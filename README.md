# PyTorch dense-GAT baseline for QM9 graph-level regression

This repository is now **a PyTorch dense-GAT baseline for QM9 graph-level regression adapted from a legacy GAT repo**.

The old TensorFlow 1.x/Cora node-classification pipeline has been replaced with a modern graph-level regression workflow built on PyTorch + PyG.

## Dependencies

Install core dependencies:

```bash
pip install -r requirements.txt
```

Required:
- `torch`
- `torch_geometric`

Optional:
- `rdkit` (some QM9 processing environments)
- `ase` (only for optional SchNet-style `qm9.db` input)

## Targets and naming

Supported target names and QM9 indices:

- `mu` -> 0
- `alpha` -> 1
- `homo` -> 2
- `lumo` -> 3
- `gap` -> 4
- `r2` -> 5
- `zpve` -> 6
- `energy_U0` -> 7
- `energy_U` -> 8
- `enthalpy_H` -> 9
- `free_G` -> 10
- `Cv` -> 11
- `atomization_U0` -> 12
- `atomization_U` -> 13
- `atomization_H` -> 14
- `atomization_G` -> 15

## Training

### Example: `energy_U0`

```bash
python train_qm9.py \
  --data_dir ./data/qm9 \
  --target energy_U0 \
  --batch_size 64 \
  --epochs 200 \
  --lr 1e-3 \
  --weight_decay 1e-6 \
  --patience 30 \
  --hidden_dim 128 \
  --num_layers 4 \
  --num_heads 4 \
  --dropout 0.1 \
  --residual \
  --seed 42 \
  --split_path splits/qm9_split.json \
  --ntrain 100000 \
  --nval 10000 \
  --ntest 10831 \
  --checkpoint_dir checkpoints \
  --use_atomref \
  --graph_mode dataset
```

### Example: `gap`

```bash
python train_qm9.py \
  --data_dir ./data/qm9 \
  --target gap \
  --batch_size 64 \
  --epochs 200 \
  --lr 1e-3 \
  --weight_decay 1e-6 \
  --patience 30 \
  --hidden_dim 128 \
  --num_layers 4 \
  --num_heads 4 \
  --dropout 0.1 \
  --residual \
  --seed 42 \
  --split_path splits/qm9_split.json \
  --ntrain 100000 \
  --nval 10000 \
  --ntest 10831 \
  --checkpoint_dir checkpoints \
  --graph_mode dataset
```

## What the pipeline does

### Default dataset path

By default, data is read from `torch_geometric.datasets.QM9` using `--data_dir` as the PyG root directory.

Optional compatibility mode: if `--data_dir` points directly to a `.db` file, the code attempts to read SchNet-style `qm9.db` via ASE.

### Dense masked batching

Even though QM9 is loaded as sparse PyG graphs, each batch is converted to dense tensors to stay close to legacy dense-GAT behavior:

- node features: `[B, N, F]`
- adjacency mask: `[B, N, N]`
- valid-node mask: `[B, N]`
- scalar targets: `[B, 1]`

`to_dense_batch` and `to_dense_adj` are used, self-loops are always included, and masks explicitly prevent padded nodes from participating in attention or pooling.

### Graph connectivity modes

- `--graph_mode dataset` (default): use QM9 graph structure from dataset edges.
- `--graph_mode cutoff --cutoff <float>`: rebuild adjacency from pairwise distances over 3D coordinates.

### Target normalization

Targets are standardized with **train-split statistics only**:

- train mean/std computed on selected target
- model trained on normalized target (or normalized residual when atomref is enabled)
- MAE is reported in original units by inverting normalization before metric computation

### Optional atomref support

With `--use_atomref`, the script attempts `dataset.atomref(target_idx)` for:

- `zpve`, `energy_U0`, `energy_U`, `enthalpy_H`, `free_G`, `Cv`

If available, training uses residual targets (`target - atomref_baseline`), then atomref is added back for validation/test metrics.
If unavailable, atomref is disabled with a warning and training continues.

## Repository layout

- `datasets/qm9_dataset.py`: target mapping, data loading, split handling, target stats, atomref helpers
- `models/gat_qm9.py`: dense multi-head GAT layers + graph-level regression head
- `train_qm9.py`: full train/val/test loop with early stopping and checkpoint restore
- `legacy/`: archived legacy TensorFlow/Cora code and artifacts
