import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.nn import functional as F

from .data import corrupt, encode, load
from .gnn import GraphClassifier, pack
from .selector import select_features


@torch.no_grad()
def evaluate(model, graphs, indices, selected, args):
    model.eval()
    probabilities, targets = [], []
    for start in range(0, len(indices), args.batch_size):
        rows = indices[start:start + args.batch_size]
        x, edges, graph_ids, y = pack(graphs, rows, selected, args.device)
        probability = model(x, edges, graph_ids, len(rows)).softmax(1)
        probabilities.append(probability.cpu().numpy())
        targets.append(y.cpu().numpy())
    probability = np.concatenate(probabilities)
    y = np.concatenate(targets)
    prediction = probability.argmax(1)
    return dict(accuracy=float(accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=range(probability.shape[1]), average="macro", zero_division=0)),
                loss=float(-np.log(probability[np.arange(len(y)), y].clip(1e-8)).mean()))


def peak_host_memory():
    if os.name == "nt":
        return float(psutil.Process().memory_info().peak_wset / 1024 ** 3)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2))


def run(args):
    if not 0 < args.retained <= 1 or not 0 < args.alpha < 1 - args.beta < 1:
        raise ValueError("Invalid retained ratio or rough approximation thresholds")
    if min(args.views, args.anchors, args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("Selector/training budgets must be positive")
    if not 0 < args.radius_quantile < 1 or not 0 < args.minimum_weight <= 1:
        raise ValueError("Invalid radius quantile or boundary weight")
    if not 0 <= args.nuisance_ratio <= 1 or not 0 <= args.label_noise <= 1:
        raise ValueError("Corruption ratios must be in [0,1]")
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    raw = load(args.dataset, args.root)
    labels = np.array([g["y"] for g in raw])
    folds = list(StratifiedKFold(10, shuffle=True, random_state=args.split_seed).split(np.zeros(len(raw)), labels))
    if not 0 <= args.fold < 10:
        raise ValueError("Fold must be between zero and nine")
    test_indices = folds[args.fold][1]
    validation_indices = folds[(args.fold + 1) % 10][1]
    train_indices = np.setdiff1d(np.arange(len(raw)), np.concatenate((test_indices, validation_indices)))
    category_train = np.concatenate([raw[i]["categories"] for i in train_indices])
    base_dimension = raw[0]["attributes"].shape[1] + sum(len(np.unique(category_train[:, c]))
                    for c in range(category_train.shape[1]))
    noise_channels = int(np.ceil(args.nuisance_ratio * base_dimension))
    graphs, schema = encode(raw, train_indices, noise_channels, args.seed + 30)
    training = [graphs[i] for i in train_indices]
    if args.label_noise:
        rng = np.random.default_rng(args.seed + 40)
        selected = rng.choice(len(training), int(len(training) * args.label_noise), replace=False)
        training = copy.deepcopy(training)
        for index in selected:
            training[index]["y"] = (training[index]["y"] + int(rng.integers(1, labels.max() + 1))) % (labels.max() + 1)
    if args.mask:
        with np.load(args.mask, allow_pickle=False) as archive:
            if not np.array_equal(archive["feature_names"], np.array(schema["names"])) or not np.array_equal(archive["train_ids"], train_indices):
                raise ValueError("A reused mask must have the same schema and training fold")
            if int(archive["seed"]) != args.seed or float(archive["nuisance_ratio"]) != args.nuisance_ratio or float(archive["label_noise"]) != args.label_noise:
                raise ValueError("A reused mask must have the same seed and corruption setup")
            result = {key: archive[key] for key in ("selected", "weights", "anchors", "signature_median", "signature_iqr")}
        result.update(selection_seconds=0.0, trace=[])
    else:
        result = select_features(training, schema, args)
    selected = result["selected"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "selection.npz", **{key: value for key, value in result.items()
                        if key not in ("trace", "selection_seconds")}, train_ids=train_indices,
                        feature_names=np.array(schema["names"]), seed=args.seed,
                        nuisance_ratio=args.nuisance_ratio, label_noise=args.label_noise)
    (output / "selection_trace.json").write_text(json.dumps(result["trace"], indent=2), encoding="utf-8")
    weights = result["weights"]
    classes = int(labels.max()) + 1
    model = GraphClassifier(len(selected), classes, args.backbone, args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    weight_tensor = torch.as_tensor(weights, device=args.device)
    best, stale = np.inf, 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(training))
        losses = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            x, edges, graph_ids, y = pack(training, rows, selected, args.device)
            optimizer.zero_grad()
            losses_per_graph = F.cross_entropy(model(x, edges, graph_ids, len(rows)), y, reduction="none")
            batch_weights = weight_tensor[rows]
            loss = (losses_per_graph * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, graphs, validation_indices, selected, args)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), validation=validation)
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best:
            best, stale = validation["loss"], 0
            torch.save(dict(state=copy.deepcopy(model.state_dict()), selected=selected.tolist(),
                            epoch=epoch, config=vars(args)), output / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    training_seconds = time.perf_counter() - start_time
    checkpoint = torch.load(output / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["state"])
    clean = evaluate(model, graphs, test_indices, selected, args)
    rows = [dict(profile="clean", rate=0, accuracy_drop=0, **clean)]
    for profile in ("missing", "measurement", "mixed", "rewire"):
        for rate in (0.1, 0.2, 0.3, 0.4):
            seed = args.seed + 2000 + int(rate * 100) + (0 if profile != "rewire" else 100)
            perturbed = corrupt(graphs, rate, profile, schema, seed)
            scores = evaluate(model, perturbed, test_indices, selected, args)
            rows.append(dict(profile=profile, rate=rate, accuracy_drop=clean["accuracy"] - scores["accuracy"], **scores))
    payload = dict(config=vars(args), dataset=args.dataset, outer_fold=args.fold, seed=args.seed,
                   selected_feature_ids=selected.tolist(), selected_feature_names=[schema["names"][i] for i in selected],
                   feature_schema=schema["names"], retained_ratio=len(selected) / len(schema["names"]),
                   selection_seconds=result["selection_seconds"], training_seconds=training_seconds,
                   parameters=sum(p.numel() for p in model.parameters()), best_epoch=checkpoint["epoch"],
                   peak_host_memory_gb=peak_host_memory(),
                   peak_accelerator_memory_gb=float(torch.cuda.max_memory_allocated(args.device) / 1024 ** 3) if str(args.device).startswith("cuda") else 0,
                   evaluation=rows, train_graph_ids=train_indices.tolist(),
                   validation_graph_ids=validation_indices.tolist(), test_graph_ids=test_indices.tolist())
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
