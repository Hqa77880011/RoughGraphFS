from pathlib import Path
from urllib.request import urlopen
import zipfile

import numpy as np


DATASETS = ("MUTAG", "PROTEINS", "NCI1", "NCI109", "DD")


def download(name, root):
    if name not in DATASETS:
        raise ValueError("Unsupported TUDataset name")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"{name}.zip"
    with urlopen(f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip", timeout=120) as response:
        with archive_path.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive path escapes the data directory")
        archive.extractall(root)


def load(name, root):
    prefix = Path(root) / name / name
    indicator = np.loadtxt(str(prefix) + "_graph_indicator.txt", dtype=int) - 1
    targets = np.loadtxt(str(prefix) + "_graph_labels.txt", dtype=int)
    _, targets = np.unique(targets, return_inverse=True)
    edges = np.loadtxt(str(prefix) + "_A.txt", delimiter=",", dtype=int, ndmin=2) - 1
    label_path = Path(str(prefix) + "_node_labels.txt")
    attribute_path = Path(str(prefix) + "_node_attributes.txt")
    categories = np.loadtxt(label_path, delimiter=",", dtype=int, ndmin=2) if label_path.exists() else np.empty((len(indicator), 0), dtype=int)
    attributes = np.loadtxt(attribute_path, delimiter=",", dtype=np.float32, ndmin=2) if attribute_path.exists() else np.empty((len(indicator), 0), dtype=np.float32)
    if len(attributes) != len(indicator) or len(categories) != len(indicator):
        raise ValueError("Node feature rows disagree with graph indicators")
    if not np.isfinite(attributes).all():
        raise ValueError("Raw attributes must be finite")
    if edges.size and (edges.min() < 0 or edges.max() >= len(indicator)):
        raise ValueError("Edge endpoints outside node IDs")
    if np.any(indicator[edges[:, 0]] != indicator[edges[:, 1]]):
        raise ValueError("An edge connects different graphs")
    graphs = []
    lookup = np.empty(len(indicator), dtype=int)
    for graph_id, target in enumerate(targets):
        nodes = np.flatnonzero(indicator == graph_id)
        lookup[nodes] = np.arange(len(nodes))
        subset = edges[indicator[edges[:, 0]] == graph_id]
        local = lookup[subset]
        local = np.unique(np.sort(local, axis=1), axis=0)
        graphs.append(dict(attributes=attributes[nodes], categories=categories[nodes],
                           edges=local, y=int(target), id=graph_id))
    return graphs


def encode(graphs, train_indices, noise_channels, seed):
    attributes = np.concatenate([graphs[i]["attributes"] for i in train_indices])
    categories = np.concatenate([graphs[i]["categories"] for i in train_indices])
    vocabularies = [np.unique(categories[:, c]) for c in range(categories.shape[1])]
    names = [f"attribute_{c}" for c in range(attributes.shape[1])]
    groups = []
    dimension = len(names)
    for column, vocabulary in enumerate(vocabularies):
        groups.append(list(range(dimension, dimension + len(vocabulary))))
        names.extend(f"label_{column}_{value}" for value in vocabulary)
        dimension += len(vocabulary)
    if dimension == 0:
        raise ValueError("No node attributes or node labels available")
    rng = np.random.default_rng(seed)
    raw = []
    for graph in graphs:
        columns = [graph["attributes"]]
        columns.extend((graph["categories"][:, c:c + 1] == vocabulary[None, :]).astype(np.float32)
                       for c, vocabulary in enumerate(vocabularies))
        x = np.concatenate(columns, 1)
        if noise_channels:
            x = np.column_stack((x, rng.normal(size=(len(x), noise_channels)).astype(np.float32)))
        raw.append(x)
    names.extend(f"nuisance_{c}" for c in range(noise_channels))
    fitted = np.concatenate([raw[i] for i in train_indices])
    mean = fitted.mean(0)
    standard = fitted.std(0).clip(1e-8)
    result = [dict(x=((x - mean) / standard).astype(np.float32), observed=np.ones(x.shape, dtype=bool),
                   edges=graph["edges"], y=graph["y"], id=graph["id"])
              for x, graph in zip(raw, graphs)]
    continuous = list(range(attributes.shape[1]))
    return result, dict(mean=mean, std=standard, groups=groups, continuous=continuous, names=names,
                        base_dimension=dimension)


def corrupt(graphs, rate, profile, schema, seed):
    rng = np.random.default_rng(seed)
    output = []
    for graph in graphs:
        x = graph["x"].copy()
        observed = graph["observed"].copy()
        edges = graph["edges"].copy()
        if profile == "mixed" and x.shape[1] > schema["base_dimension"]:
            begin = schema["base_dimension"]
            x[:, begin:] = (rng.normal(size=x[:, begin:].shape) - schema["mean"][begin:]) / schema["std"][begin:]
        if profile in ("measurement", "mixed"):
            for group in schema["groups"]:
                if len(group) < 2:
                    continue
                rows = np.flatnonzero(rng.random(len(x)) < rate)
                unscaled = x[:, group] * schema["std"][group] + schema["mean"][group]
                category = unscaled.argmax(1)
                replacement = (category[rows] + rng.integers(1, len(group), len(rows))) % len(group)
                unscaled[rows] = 0
                unscaled[rows, replacement] = 1
                x[:, group] = (unscaled - schema["mean"][group]) / schema["std"][group]
            if schema["continuous"]:
                values = x[:, schema["continuous"]]
                changed = rng.random(values.shape) < rate
                x[:, schema["continuous"]] = values + changed * rng.normal(size=values.shape)
        if profile in ("missing", "mixed"):
            missing = rng.random(x.shape) < rate
            x[missing] = 0
            observed[missing] = False
        if profile == "rewire" and len(edges):
            count = int(len(edges) * rate)
            chosen = rng.choice(len(edges), count, replace=False)
            edges[chosen, 1] = rng.integers(0, len(x), count)
        output.append(dict(x=x, observed=observed, edges=edges, y=graph["y"], id=graph["id"]))
    return output
