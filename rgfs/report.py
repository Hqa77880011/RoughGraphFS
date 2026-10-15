import json
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def holm(rows):
    previous = 0
    for index, row in enumerate(sorted(rows, key=lambda value: value["pvalue"])):
        previous = max(previous, min(1, (len(rows) - index) * row["pvalue"]))
        row["holm_pvalue"] = previous


def summarize(paths, output, resamples=10000):
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    identities = [(run["dataset"], run["outer_fold"], run["seed"], run["config"]["selector"], run["config"]["backbone"]) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("Supply only one selected configuration per dataset/fold/seed/selector/backbone")
    aggregate = []
    stability = []
    for dataset in sorted({run["dataset"] for run in runs}):
        for method in sorted({run["config"]["selector"] for run in runs}):
            for backbone in sorted({run["config"]["backbone"] for run in runs}):
                selected_runs = [run for run in runs if run["dataset"] == dataset
                                 and run["config"]["selector"] == method and run["config"]["backbone"] == backbone]
                if not selected_runs:
                    continue
                sets = [set(run["selected_feature_names"]) for run in selected_runs]
                universe = sorted(set().union(*(set(run["feature_schema"]) for run in selected_runs)))
                masks = np.array([[name in selected for name in universe] for selected in sets], dtype=float)
                pairwise = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
                fraction = masks.mean()
                nogueira = 1 - masks.var(axis=0, ddof=1).mean() / (fraction * (1 - fraction)) if len(masks) > 1 and 0 < fraction < 1 else None
                stability.append(dict(dataset=dataset, selector=method, backbone=backbone,
                                      pairwise_jaccard=float(np.mean(pairwise)) if pairwise else None,
                                      nogueira=float(nogueira) if nogueira is not None else None, runs=len(sets)))
                conditions = {(row["profile"], row["rate"]) for run in selected_runs for row in run["evaluation"]}
                for profile, rate in sorted(conditions):
                    rows = [(run, next(row for row in run["evaluation"] if row["profile"] == profile and row["rate"] == rate))
                            for run in selected_runs]
                    fold_scores = {}
                    for run, row in rows:
                        fold_scores.setdefault(run["outer_fold"], []).append(row["accuracy"])
                    aggregate.append(dict(dataset=dataset, selector=method, backbone=backbone, profile=profile, rate=rate,
                                          accuracy=float(np.mean([row["accuracy"] for _, row in rows])),
                                          accuracy_sd=float(np.std([row["accuracy"] for _, row in rows], ddof=1)) if len(rows) > 1 else None,
                                          macro_f1=float(np.mean([row["macro_f1"] for _, row in rows])),
                                          accuracy_drop=float(np.mean([row["accuracy_drop"] for _, row in rows])),
                                          fold_mean_accuracy={str(fold): float(np.mean(values)) for fold, values in fold_scores.items()},
                                          fold_seeds={str(fold): sorted(run["seed"] for run, _ in rows if run["outer_fold"] == fold) for fold in fold_scores},
                                          selection_seconds=float(np.mean([run["selection_seconds"] for run, _ in rows])),
                                          training_seconds=float(np.mean([run["training_seconds"] for run, _ in rows])),
                                          peak_host_memory_gb=max(run["peak_host_memory_gb"] for run, _ in rows),
                                          peak_accelerator_memory_gb=max(run["peak_accelerator_memory_gb"] for run, _ in rows),
                                          parameters=sorted({run["parameters"] for run, _ in rows})))
    paired = []
    rng = np.random.default_rng(0)
    for reference in [row for row in aggregate if row["selector"] == "rgfs"]:
        for baseline in aggregate:
            if baseline["selector"] == "rgfs" or any(reference[key] != baseline[key] for key in ("dataset", "backbone", "profile", "rate")):
                continue
            common = sorted(set(reference["fold_mean_accuracy"]) & set(baseline["fold_mean_accuracy"]))
            if len(common) < 2:
                continue
            if any(reference["fold_seeds"][fold] != baseline["fold_seeds"][fold] for fold in common):
                raise ValueError("Paired outer-fold means require matching network seeds")
            difference = np.array([reference["fold_mean_accuracy"][fold] - baseline["fold_mean_accuracy"][fold] for fold in common])
            boot = np.median(rng.choice(difference, (resamples, len(difference)), replace=True), axis=1)
            paired.append(dict(dataset=reference["dataset"], backbone=reference["backbone"], profile=reference["profile"],
                               rate=reference["rate"], baseline=baseline["selector"], folds=common,
                               median_difference=float(np.median(difference)), ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                               pvalue=float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0))
    families = {(row["dataset"], row["backbone"], row["profile"], row["rate"]) for row in paired}
    for family in families:
        holm([row for row in paired if (row["dataset"], row["backbone"], row["profile"], row["rate"]) == family])
    friedman = []
    dataset_pairs = []
    for backbone in sorted({row["backbone"] for row in aggregate}):
        rows = [row for row in aggregate if row["backbone"] == backbone and row["profile"] == "mixed" and row["rate"] == 0.3]
        datasets = sorted({row["dataset"] for row in rows})
        methods = sorted({row["selector"] for row in rows})
        table = {(row["dataset"], row["selector"]): row["accuracy"] for row in rows}
        if len(datasets) >= 2 and len(methods) >= 3 and all((dataset, method) in table for dataset in datasets for method in methods):
            values = np.array([[table[dataset, method] for method in methods] for dataset in datasets])
            result = friedmanchisquare(*values.T)
            ranks = np.array([rankdata(-row) for row in values]).mean(0)
            friedman.append(dict(backbone=backbone, datasets=datasets, methods=methods,
                                 statistic=float(result.statistic) if np.isfinite(result.statistic) else None,
                                 pvalue=float(result.pvalue) if np.isfinite(result.pvalue) else None,
                                 mean_ranks=dict(zip(methods, ranks.tolist()))))
            if "rgfs" in methods:
                reference = values[:, methods.index("rgfs")]
                family = []
                for method in methods:
                    if method == "rgfs":
                        continue
                    difference = reference - values[:, methods.index(method)]
                    family.append(dict(backbone=backbone, baseline=method, datasets=datasets,
                                       mean_difference=float(difference.mean()),
                                       pvalue=float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0))
                holm(family)
                dataset_pairs.extend(family)
    for name, value in (("summary", aggregate), ("stability", stability), ("paired", paired), ("friedman", friedman), ("dataset_paired", dataset_pairs)):
        (destination / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads(Path(summary).read_text(encoding="utf-8"))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({row["dataset"] for row in rows}):
        for backbone in sorted({row["backbone"] for row in rows}):
            for profile in ("missing", "measurement", "mixed", "rewire"):
                subset = [row for row in rows if row["dataset"] == dataset and row["backbone"] == backbone and row["profile"] == profile]
                if not subset:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                for method in sorted({row["selector"] for row in subset}):
                    points = sorted([row for row in subset if row["selector"] == method], key=lambda row: row["rate"])
                    ax.errorbar([row["rate"] for row in points], [row["accuracy"] for row in points],
                                yerr=[row["accuracy_sd"] or 0 for row in points], marker="o", label=method)
                ax.set(xlabel="Corruption rate", ylabel="Accuracy", title=f"{dataset} / {backbone} / {profile}")
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(destination / f"{dataset}_{backbone}_{profile}.png", dpi=180)
                plt.close(fig)
