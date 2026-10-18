# RoughGraphFS

Implementation of **RoughGraphFS: Topology-Conditioned Rough Dependency for Feature Selection under Graph Attribute Corruption**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

The supported datasets are MUTAG, PROTEINS, NCI1, NCI109 and DD from [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/).

Alternatively place the released TU text files in `data/<dataset>/`. The loader reads graph indicators, graph labels, undirected edges, optional node attributes and optional node labels. It checks that edges stay inside graph boundaries. Undirected edges are deduplicated for signatures and expanded into both directions for message passing.

Continuous node attributes are followed by one-hot node-label channels. Category vocabularies, feature means and standard deviations are fitted using training graphs only. Unknown validation/test categories map to an all-zero raw one-hot group. Node features are standardized with training statistics; missing entries are then represented by zero plus a separate observation mask for signature computation.

```bash
python -m rgfs download --dataset MUTAG --root data
```

## Training and inference

```bash
python -m rgfs train --dataset MUTAG --root data --output outputs/rgfs_gin_f0_s13 --fold 0 --seed 13 --selector rgfs --backbone gin --device cpu
```

## Evaluation

```bash
python -m rgfs summarize --metrics outputs/rgfs_gin_f0_s13/metrics.json --output outputs/statistics
python -m rgfs plot --summary outputs/statistics/summary.json --output outputs/figures
```

## Outputs

Each run writes `selection.npz`, `selection_trace.json`, `best.pt`, `history.json` and `metrics.json`. Summaries/figures read those saved records. One-fold summaries are descriptive; paired inference requires matching independent folds.

Source inspection: Python 3.13.5, torch 2.7.1+cu118, numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
