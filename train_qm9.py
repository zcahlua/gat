from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from datasets.qm9_dataset import (
    compute_target_stats,
    get_atomref_vector,
    get_qm9_data,
    get_target_index,
    make_splits,
)
from models.gat_qm9 import GATQM9Regressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense-GAT QM9 graph-level regression")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--residual", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_path", type=str, default="splits/qm9_split.json")
    p.add_argument("--ntrain", type=int, default=100000)
    p.add_argument("--nval", type=int, default=10000)
    p.add_argument("--ntest", type=int, default=10831)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--use_atomref", action="store_true")
    p.add_argument("--graph_mode", type=str, choices=["dataset", "cutoff"], default="dataset")
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--loss", type=str, choices=["mse", "huber"], default="mse")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dense_inputs(batch, graph_mode: str, cutoff: float, atomref: Optional[torch.Tensor]):
    x_dense, node_mask = to_dense_batch(batch.x, batch.batch)
    z_dense, _ = to_dense_batch(batch.z.unsqueeze(-1).float(), batch.batch)
    z_dense = z_dense.squeeze(-1).long()

    if graph_mode == "dataset" and hasattr(batch, "edge_index") and batch.edge_index is not None:
        adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=x_dense.size(1)).bool()
    else:
        pos_dense, _ = to_dense_batch(batch.pos, batch.batch)
        dists = torch.cdist(pos_dense, pos_dense)
        adj = dists <= cutoff

    valid_pairs = node_mask.unsqueeze(-1) & node_mask.unsqueeze(1)
    eye = torch.eye(adj.size(-1), device=adj.device, dtype=torch.bool).unsqueeze(0)
    adj = (adj | eye) & valid_pairs

    baseline = None
    if atomref is not None:
        atomref = atomref.to(x_dense.device)
        baseline = atomref[z_dense] * node_mask
        baseline = baseline.sum(dim=1, keepdim=True)

    return x_dense, adj, node_mask, baseline


def get_targets(batch, target_idx: int) -> torch.Tensor:
    return batch.y[:, target_idx].view(-1, 1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    target_idx: int,
    target_mean: float,
    target_std: float,
    graph_mode: str,
    cutoff: float,
    atomref: Optional[torch.Tensor],
    loss_name: str,
) -> Tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    loss_total = 0.0
    mae_total = 0.0
    count = 0

    for batch in loader:
        batch = batch.to(device)
        x_dense, adj, node_mask, baseline = make_dense_inputs(batch, graph_mode, cutoff, atomref)
        y = get_targets(batch, target_idx)
        if baseline is None:
            baseline = torch.zeros_like(y)

        y_residual = y - baseline
        y_norm = (y_residual - target_mean) / target_std

        pred_norm, _ = model(x_dense, adj, node_mask)

        if loss_name == "mse":
            loss = F.mse_loss(pred_norm, y_norm)
        else:
            loss = F.huber_loss(pred_norm, y_norm, delta=1.0)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        pred = pred_norm * target_std + target_mean + baseline
        mae = (pred - y).abs().mean()

        bs = y.size(0)
        loss_total += loss.item() * bs
        mae_total += mae.item() * bs
        count += bs

    return loss_total / count, mae_total / count


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset = get_qm9_data(args.data_dir)
    target_idx = get_target_index(args.target)
    split = make_splits(len(dataset), args.seed, args.ntrain, args.nval, args.ntest, args.split_path)

    train_ds = dataset[split["train"]]
    val_ds = dataset[split["val"]]
    test_ds = dataset[split["test"]]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    mean, std = compute_target_stats(dataset, split["train"], target_idx)
    print(f"Target={args.target} idx={target_idx} train_mean={mean:.6f} train_std={std:.6f}")

    atomref = get_atomref_vector(dataset, args.target, target_idx) if args.use_atomref else None
    if args.use_atomref and atomref is None:
        print("[warn] atomref requested but unavailable; continuing without atomref")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = dataset[0].x.size(-1)
    model = GATQM9Regressor(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        residual=args.residual,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_{args.target}.pt"

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            target_idx,
            mean,
            std,
            args.graph_mode,
            args.cutoff,
            atomref,
            args.loss,
        )
        val_loss, val_mae = run_epoch(
            model,
            val_loader,
            None,
            device,
            target_idx,
            mean,
            std,
            args.graph_mode,
            args.cutoff,
            atomref,
            args.loss,
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} train_mae={train_mae:.5f} "
            f"| val_loss={val_loss:.5f} val_mae={val_mae:.5f}"
        )

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch} val_mae={best_val:.5f}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_mae = run_epoch(
        model,
        test_loader,
        None,
        device,
        target_idx,
        mean,
        std,
        args.graph_mode,
        args.cutoff,
        atomref,
        args.loss,
    )
    print(f"Test | loss={test_loss:.5f} mae={test_mae:.5f} | checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
