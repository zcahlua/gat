from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import QM9

TARGET_MAP: Dict[str, int] = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "zpve": 6,
    "energy_U0": 7,
    "energy_U": 8,
    "enthalpy_H": 9,
    "free_G": 10,
    "Cv": 11,
    "atomization_U0": 12,
    "atomization_U": 13,
    "atomization_H": 14,
    "atomization_G": 15,
}

ATOMREF_TARGETS = {"zpve", "energy_U0", "energy_U", "enthalpy_H", "free_G", "Cv"}


def _build_split_indices(
    dataset_len: int,
    seed: int,
    ntrain: int,
    nval: int,
    ntest: int,
    split_path: Optional[str],
) -> Dict[str, List[int]]:
    if split_path and Path(split_path).exists():
        with Path(split_path).open("r", encoding="utf-8") as f:
            split = json.load(f)
        return {k: list(v) for k, v in split.items()}

    total = ntrain + nval + ntest
    if total > dataset_len:
        raise ValueError(f"Requested {total} samples, but dataset has only {dataset_len}.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(dataset_len).tolist()
    split = {
        "train": perm[:ntrain],
        "val": perm[ntrain : ntrain + nval],
        "test": perm[ntrain + nval : total],
    }
    if split_path:
        path = Path(split_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
    return split


def _load_qm9_from_db(db_path: str) -> List[Data]:
    try:
        from ase.db import connect
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ASE is required for reading qm9.db files.") from exc

    data_list: List[Data] = []
    with connect(db_path) as db:
        for row in db.select():
            atoms = row.toatoms()
            z = torch.tensor(atoms.numbers, dtype=torch.long)
            pos = torch.tensor(atoms.positions, dtype=torch.float)

            y_values = []
            for name in TARGET_MAP:
                y_values.append(float(row.data.get(name, 0.0)))
            y = torch.tensor(y_values, dtype=torch.float).view(1, -1)

            data_list.append(Data(z=z, pos=pos, y=y))

    if not data_list:
        raise RuntimeError(f"No molecules found in {db_path}.")
    return data_list


def get_qm9_data(data_dir: str):
    path = Path(data_dir)
    if path.is_file() and path.suffix == ".db":
        warnings.warn("Using optional ASE qm9.db reader; default path is torch_geometric.datasets.QM9.")
        return _load_qm9_from_db(str(path))
    return QM9(root=str(path))


def get_target_index(target: str) -> int:
    if target not in TARGET_MAP:
        choices = ", ".join(TARGET_MAP.keys())
        raise ValueError(f"Unknown target '{target}'. Valid targets: {choices}")
    return TARGET_MAP[target]


def make_splits(
    dataset_len: int,
    seed: int,
    ntrain: int,
    nval: int,
    ntest: int,
    split_path: Optional[str],
) -> Dict[str, List[int]]:
    return _build_split_indices(dataset_len, seed, ntrain, nval, ntest, split_path)


def compute_target_stats(dataset, indices: Sequence[int], target_idx: int) -> Tuple[float, float]:
    values = []
    for idx in indices:
        values.append(float(dataset[idx].y.view(-1)[target_idx]))
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-12:
        std = 1.0
    return mean, std


def get_atomref_vector(dataset, target: str, target_idx: int) -> Optional[torch.Tensor]:
    if target not in ATOMREF_TARGETS:
        return None

    if hasattr(dataset, "atomref"):
        atomref = dataset.atomref(target_idx)
        if atomref is not None:
            return atomref.view(-1)

    warnings.warn(f"Atomref unavailable for target={target}; disabling atomref correction.")
    return None


def molecule_atomref_baseline(z: torch.Tensor, atomref: torch.Tensor) -> torch.Tensor:
    # z: [N], atomref indexed by atomic number
    max_z = int(z.max().item())
    if atomref.numel() <= max_z:
        pad = torch.zeros(max_z + 1 - atomref.numel(), device=atomref.device, dtype=atomref.dtype)
        atomref = torch.cat([atomref, pad], dim=0)
    return atomref[z].sum()
