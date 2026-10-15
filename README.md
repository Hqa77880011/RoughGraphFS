# RoughGraphFS

Implementation of **RoughGraphFS: Topology-Conditioned Rough Dependency for Feature Selection under Graph Attribute Corruption**.

RGFS selects original node-feature channels using graph-level robust signatures, anchor-based rough dependency, redundancy and perturbation stability. Rough-boundary memberships weight training graphs. At inference, a GNN consumes only the selected node attributes; no rough computation or test labels are required.

## Install

Use Python 3.11 or 3.12 from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate `.venv` before installation (`source .venv/bin/activate` on Linux/macOS, `.venv\Scripts\Activate.ps1` in PowerShell). Install a suitable CUDA PyTorch build first for GPU training. The message-passing code uses native PyTorch indexing and reduction operations and does not require PyTorch Geometric or compiled graph extensions.

## 1. Download graph data

The supported datasets are MUTAG, PROTEINS, NCI1, NCI109 and DD from [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/).

```bash
python -m rgfs download --dataset MUTAG --root data
python -m rgfs download --dataset PROTEINS --root data
```

Alternatively place the released TU text files in `data/<dataset>/`. The loader reads graph indicators, graph labels, undirected edges, optional node attributes and optional node labels. It checks that edges stay inside graph boundaries. Undirected edges are deduplicated for signatures and expanded into both directions for message passing.

Continuous node attributes are followed by one-hot node-label channels. Category vocabularies, feature means and standard deviations are fitted using training graphs only. Unknown validation/test categories map to an all-zero raw one-hot group. Node features are standardized with training statistics; missing entries are then represented by zero plus a separate observation mask for signature computation.

## 2. Fit a fold-local selector and GNN

```bash
python -m rgfs train --dataset MUTAG --root data --output outputs/rgfs_gin_f0_s13 --fold 0 --seed 13 --selector rgfs --backbone gin --device cuda
```

The split is ten-fold stratified with `--split-seed 2024`. Fold `f` is test, `(f+1)%10` is validation, and the other eight folds fit the selector and GNN. Every compared method must use the same split seed and network/selector seeds. The test fold is hidden from scaling, anchors, feature ranking, boundary weights and checkpoint selection.

Each feature's signature contains median, IQR, lower/upper quartiles, graph total variation and observed fraction. Total variation averages absolute standardized feature differences over the original undirected edge count, excluding edges whose endpoints both lack the channel. All six coordinates are median/IQR normalized from training graphs. Distances average absolute coordinate differences and are clipped at one.

At most 512 class-stratified training anchors define triangular-kernel rough neighborhoods. Dependency is the fraction of anchors whose maximum class membership reaches `1-beta`. The greedy criterion maximizes clean dependency gain minus maximum signature correlation and the change in that gain across corrupted training views. Final boundary weights interpolate from `minimum_weight` to one using the maximum class membership.

| Argument | Default | Meaning |
|---|---:|---|
| `--retained` | 0.5 | Fraction of observed-schema channels retained, rounded up |
| `--anchors`, `--views` | 512, 3 | Anchor cap and independent corruption views |
| `--radius-quantile` | 0.2 | Quantile of clean all-channel anchor distances |
| `--alpha`, `--beta` | 0.1, 0.1 | Upper/lower rough thresholds |
| `--redundancy`, `--stability` | 0.05, 0.05 | Redundancy and instability coefficients |
| `--minimum-weight` | 0.5 | Positive training-graph weight floor |
| `--inner-rate` | 0.1 | Corruption severity for selector views |
| `--nuisance-ratio` | 0.3 | Number of Gaussian nuisance channels divided by base channel count |
| `--label-noise` | 0 | Fraction of fitting labels flipped for the label-noise stress test |
| `--hidden` | 64 | GNN hidden width |
| `--epochs`, `--patience` | 300, 50 | Maximum epochs and validation-loss patience |
| `--batch-size` | 32 | Graphs per training batch; use 16 for DD |
| `--lr`, `--weight-decay` | 0.001, 0.0001 | AdamW settings |

If a rough neighborhood has zero total kernel weight, the nearest anchor receives unit weight. This makes the paper's nearest-anchor fallback actionable even when that anchor is outside the nominal radius. Constant signature coordinates use a small scale floor. The signature dimension is consistently six, including observed fraction.

## 3. Baselines and ablations

```bash
python -m rgfs train --dataset MUTAG --output outputs/all_gin_f0_s13 --fold 0 --seed 13 --selector all
python -m rgfs train --dataset MUTAG --output outputs/mrmr_gin_f0_s13 --fold 0 --seed 13 --selector mrmr
python -m rgfs train --dataset MUTAG --output outputs/no_stability_f0_s13 --fold 0 --seed 13 --selector no_stability
```

Implemented selectors:

| Selector | Definition |
|---|---|
| `rgfs` | Full rough dependency, redundancy, perturbation stability and boundary weights |
| `all` | Every available feature channel |
| `random` | Seeded random channels at the same retained budget |
| `variance` | Mean normalized-signature variance ranking |
| `mrmr` | Mean coordinate mutual information minus maximum channel-signature correlation |
| `relieff` | Ten nearest hits/misses under channel-signature distances, with multiclass prior weighting |
| `no_topology` | Removes total variation from the signature |
| `no_stability` | Sets perturbation penalty to zero |
| `no_redundancy` | Sets redundancy penalty to zero |
| `no_boundary` | Uses uniform graph weights after full mask selection |

The repository includes the listed transparent mask baselines; it does not rename them as LSDFS, GAEFS or native pooling architectures. For mRMR, the relevance score averages mutual information across a channel's six signature coordinates; redundancy uses the same normalized signature correlation as RGFS. ReliefF uses all training graphs as evaluation points and excludes self from nearest hits.

## 4. Transfer a mask across backbones

Native PyTorch implementations provide GIN, GCN, single-head GAT and GraphSAGE. Each has three message-passing layers, batch normalization, ReLU, global mean pooling and classifier dropout 0.5.

```bash
python -m rgfs train --dataset MUTAG --output outputs/rgfs_gcn_f0_s13 --fold 0 --seed 13 --backbone gcn --mask outputs/rgfs_gin_f0_s13/selection.npz
```

`--mask` reuses feature IDs and graph weights without re-fitting the selector. It checks the training fold, channel names, random seed and corruption setup. Keep `--nuisance-ratio` and `--label-noise` identical to the source run. Parameter counts are measured for every run. Different backbone formulas have different counts at the same hidden width; adjust `--hidden` when constructing a parameter-matched comparison and report the stored counts.

The GIN has a learned epsilon and a two-layer MLP at each step; GCN uses symmetric self-loop normalization; GraphSAGE concatenates self and mean-neighbor features; GAT normalizes learned edge attention per target node. Rough weights affect the loss and never change message passing.

## 5. Attribute and stress evaluation

The selected checkpoint is evaluated on clean test features and missing, measurement, mixed and edge-rewire profiles at rates 0.1/0.2/0.3/0.4. Missingness zeroes standardized entries. Measurement perturbation replaces a categorical value with another training-vocabulary category or adds unit-scale Gaussian noise to selected standardized continuous entries. Mixed corruption performs measurement changes followed by independent missingness. Rewiring changes sampled edge destinations and preserves the stored edge count.

Nuisance channels are appended once per run, before selector fitting, with count `ceil(nuisance_ratio * base_channels)`. They remain present in the clean and corrupted evaluations so input dimensionality and the fitted mask stay fixed. Thus `accuracy_drop` compares clean/noisy observations under the same nuisance schema. For the irrelevant-channel curve, run separate fits with `--nuisance-ratio 0`, `0.1`, `0.2`, `0.3`, `0.4`; compare their clean rows. To study the paper's mixed setting at a particular severity, match the run's nuisance ratio to that severity and read its corresponding mixed row. Selectors fit clean augmented features and generate their declared inner corruption views; GNN fitting uses those clean augmented graphs.

The label-noise option changes only fitting labels. Validation/test labels remain the ground truth. Keep this stress analysis separate from the clean-label method comparison. Training graph IDs in output make the affected partition explicit.

## 6. Select configurations and repeat folds

Use folds 0 through 9 and five matched seeds. Within each outer fold, select configurations on validation loss only. The paper's GNN search uses learning rates 0.001/0.003 and weight decay 0/0.0001/0.001. The selector grid is retained fraction 0.25/0.5/0.75; views 3/5; radius quantile 0.1/0.2/0.3; alpha/beta 0.1/0.2; redundancy/stability 0/0.05/0.1; minimum weight 0.25/0.5/0.75; inner rate 0.1/0.2. Sample at most 40 configurations per selector within a fold before observing test results.

The CLI exposes these parameters rather than launching the entire search automatically. A single training command is one explicit fold/seed/configuration job. `best.pt` is selected by validation cross-entropy with patience; `history.json` preserves the validation progression needed to choose among jobs.

## 7. Results, stability and statistics

Each run writes:

- `selection.npz`: selected channel IDs, weights, training IDs, schema, anchor IDs and signature scaling.
- `selection_trace.json`: per-step dependency, redundancy, instability and chosen score.
- `best.pt`: the chosen GNN state and configuration.
- `history.json`: training loss and validation loss/accuracy/F1.
- `metrics.json`: clean/noisy accuracy, Macro-F1, clean-minus-noisy drop, retained ratio, selected semantic feature names, selector/training time, peak host/accelerator memory, parameter count and fold memberships.

Peak host memory is the operating system's process high-water mark; accelerator memory is PyTorch's peak allocated memory during the job. Times cover selector fitting and the actual early-stopped GNN training run separately. A reused mask reports zero selection time for that transfer job.

```bash
python -m rgfs summarize --metrics outputs/rgfs_gin_f0_s13/metrics.json outputs/all_gin_f0_s13/metrics.json --output outputs/statistics
python -m rgfs plot --summary outputs/statistics/summary.json --output outputs/figures
```

Supply one validation-selected configuration per dataset/fold/seed/selector/backbone. Summaries calculate descriptive means and sample standard deviations, pairwise mask Jaccard and the Nogueira estimator over semantic channel names. The latter can be negative when variability exceeds independent selection, and is undefined for fewer than two masks or a degenerate retain-all/retain-none budget.

For paired inference, seeds are averaged inside each outer fold; matched seed lists are required across compared selectors. At least two matched folds are needed for Wilcoxon tests and 10,000 fold-bootstrap intervals for the median accuracy difference. Holm correction is applied across baselines within each dataset/backbone/condition. Complete multi-dataset tables additionally produce Friedman ranks/tests and Holm-adjusted RGFS pairwise comparisons over dataset means. Incomplete statistical families produce no invented significance values.

Plots read generated summaries and show corruption curves with descriptive seed/fold variability. No monotonicity is imposed on measured curves.
