import time

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from .data import corrupt


SELECTORS = ("rgfs", "all", "random", "variance", "mrmr", "relieff", "no_topology",
             "no_stability", "no_redundancy", "no_boundary")


def signatures(graphs, topology=True):
    output = []
    for graph in graphs:
        x, observed, edges = graph["x"], graph["observed"], graph["edges"]
        signature = np.zeros((x.shape[1], 6), dtype=np.float32)
        for channel in range(x.shape[1]):
            values = x[observed[:, channel], channel]
            if len(values):
                q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
                signature[channel, :4] = [median, q75 - q25, q25, q75]
            if len(edges) and topology:
                valid = observed[edges[:, 0], channel] | observed[edges[:, 1], channel]
                pairs = edges[valid]
                if len(pairs):
                    signature[channel, 4] = np.abs(x[pairs[:, 0], channel] - x[pairs[:, 1], channel]).sum() / max(1, len(edges))
            signature[channel, 5] = observed[:, channel].mean()
        output.append(signature)
    return np.stack(output)


def anchor_indices(labels, cap, seed):
    rng = np.random.default_rng(seed)
    if len(labels) <= cap:
        return np.arange(len(labels))
    classes, counts = np.unique(labels, return_counts=True)
    if cap < len(classes):
        raise ValueError("Anchor cap is smaller than the class count")
    ideal = counts / counts.sum() * (cap - len(classes))
    allocated = np.floor(ideal).astype(int) + 1
    for i in np.argsort(-(ideal - np.floor(ideal)), kind="stable")[:cap - allocated.sum()]:
        allocated[i] += 1
    output = []
    for label, count in zip(classes, allocated):
        rows = np.flatnonzero(labels == label)
        output.extend(rng.choice(rows, min(count, len(rows)), replace=False))
    if len(output) < cap:
        available = np.setdiff1d(np.arange(len(labels)), output)
        output.extend(rng.choice(available, cap - len(output), replace=False))
    return np.asarray(sorted(output))


def distances(first, second):
    result = np.empty((first.shape[1], len(first), len(second)), dtype=np.float32)
    for channel in range(first.shape[1]):
        result[channel] = np.minimum(1, np.abs(first[:, None, channel] - second[None, :, channel]).mean(-1))
    return result


def membership(distance, labels, radius, classes):
    kernel = np.maximum(0, 1 - distance / (radius + 1e-8))
    empty = kernel.sum(1) <= 1e-8
    kernel[empty] = 0
    kernel[np.flatnonzero(empty), distance[empty].argmin(1)] = 1
    return kernel @ np.eye(classes)[labels] / kernel.sum(1, keepdims=True)


def select_features(graphs, schema, args):
    start = time.perf_counter()
    labels = np.array([g["y"] for g in graphs])
    dimension = graphs[0]["x"].shape[1]
    count = max(1, min(dimension, int(np.ceil(args.retained * dimension))))
    rng = np.random.default_rng(args.seed)
    clean = signatures(graphs, args.selector != "no_topology")
    median = np.median(clean, axis=0)
    iqr = (np.quantile(clean, 0.75, axis=0) - np.quantile(clean, 0.25, axis=0)).clip(1e-8)
    clean = (clean - median) / iqr
    anchors = anchor_indices(labels, args.anchors, args.seed)
    reference = clean[anchors]
    vectorized = reference.transpose(1, 0, 2).reshape(dimension, -1)
    centered = vectorized - vectorized.mean(1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1).clip(1e-8)
    correlation = np.abs(centered @ centered.T / (norm[:, None] * norm[None, :]))
    weights = np.ones(len(graphs), dtype=np.float32)
    trace = []
    if args.selector == "all":
        selected = list(range(dimension))
    elif args.selector == "random":
        selected = rng.choice(dimension, count, replace=False).tolist()
    elif args.selector == "variance":
        selected = np.argsort(-clean.var(axis=(0, 2)), kind="stable")[:count].tolist()
    elif args.selector == "mrmr":
        flattened = clean.reshape(len(clean), -1)
        relevance = mutual_info_classif(flattened, labels, random_state=args.seed).reshape(dimension, 6).mean(1)
        selected = []
        for _ in range(count):
            penalty = correlation[:, selected].max(1) if selected else 0
            score = relevance - args.redundancy * penalty
            score[selected] = -np.inf
            selected.append(int(score.argmax()))
    elif args.selector == "relieff":
        importance = np.zeros(dimension)
        for row in range(len(graphs)):
            per_channel = np.minimum(1, np.abs(clean - clean[row:row + 1]).mean(-1)).T
            combined = per_channel.mean(0)
            hits = np.flatnonzero(labels == labels[row])
            hits = hits[hits != row]
            if not len(hits):
                raise ValueError("ReliefF needs at least two training graphs per class")
            near_hits = hits[np.argsort(combined[hits])[:10]]
            importance -= per_channel[:, near_hits].mean(1) / len(graphs)
            for label in np.unique(labels):
                if label == labels[row]:
                    continue
                misses = np.flatnonzero(labels == label)
                near_misses = misses[np.argsort(combined[misses])[:10]]
                prior = np.mean(labels == label) / (1 - np.mean(labels == labels[row]))
                importance += prior * per_channel[:, near_misses].mean(1) / len(graphs)
        selected = np.argsort(-importance, kind="stable")[:count].tolist()
    else:
        views = [reference]
        for view in range(args.views):
            perturbed = corrupt(graphs, args.inner_rate, "mixed", schema, args.seed + 1000 + view)
            value = (signatures(perturbed, args.selector != "no_topology") - median) / iqr
            views.append(value[anchors])
        cached = [distances(view, view) for view in views]
        pair_distance = cached[0].mean(0)
        upper = pair_distance[np.triu_indices(len(anchors), 1)]
        radius = max(1e-6, float(np.quantile(upper, args.radius_quantile))) if len(upper) else 1e-6
        sums = [np.zeros_like(pair_distance) for _ in views]
        dependencies = np.zeros(len(views))
        selected = []
        classes = int(labels.max()) + 1
        for size in range(count):
            best = None
            for channel in range(dimension):
                if channel in selected:
                    continue
                new = []
                for view in range(len(views)):
                    distance = (sums[view] + cached[view][channel]) / (size + 1)
                    mu = membership(distance, labels[anchors], radius, classes)
                    new.append(np.mean(mu.max(1) >= 1 - args.beta))
                gain = np.asarray(new) - dependencies
                redundancy = correlation[channel, selected].max() if selected else 0
                instability = np.mean(np.abs(gain[1:] - gain[0]))
                lam = 0 if args.selector == "no_redundancy" else args.redundancy
                eta = 0 if args.selector == "no_stability" else args.stability
                score = gain[0] - lam * redundancy - eta * instability
                if best is None or score > best[0]:
                    best = (score, channel, np.asarray(new), float(instability), float(redundancy))
            _, channel, dependencies, instability, redundancy = best
            selected.append(channel)
            for view in range(len(views)):
                sums[view] += cached[view][channel]
            trace.append(dict(channel=channel, score=float(best[0]), dependency=float(dependencies[0]),
                              instability=instability, redundancy=redundancy))
        final_distance = distances(clean[:, selected], reference[:, selected]).mean(0)
        mu = membership(final_distance, labels[anchors], radius, classes)
        weights = args.minimum_weight + (1 - args.minimum_weight) * np.clip(
            (mu.max(1) - args.alpha) / (1 - args.alpha), 0, 1)
        if args.selector == "no_boundary":
            weights = np.ones(len(graphs), dtype=np.float32)
    return dict(selected=np.array(selected), weights=weights.astype(np.float32),
                selection_seconds=time.perf_counter() - start, trace=trace,
                anchors=anchors, signature_median=median, signature_iqr=iqr)
