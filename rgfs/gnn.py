import torch
from torch import nn
from torch.nn import functional as F


BACKBONES = ("gin", "gcn", "gat", "sage")


class Layer(nn.Module):
    def __init__(self, inputs, hidden, backbone):
        super().__init__()
        self.backbone = backbone
        if backbone == "gin":
            self.epsilon = nn.Parameter(torch.zeros(()))
            self.transform = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        elif backbone == "sage":
            self.transform = nn.Linear(inputs * 2, hidden)
        else:
            self.transform = nn.Linear(inputs, hidden, bias=False)
            self.bias = nn.Parameter(torch.zeros(hidden))
        if backbone == "gat":
            self.source_attention = nn.Parameter(torch.empty(hidden))
            self.target_attention = nn.Parameter(torch.empty(hidden))
            nn.init.normal_(self.source_attention, std=hidden ** -0.5)
            nn.init.normal_(self.target_attention, std=hidden ** -0.5)

    def forward(self, x, edges):
        source, target = edges
        if self.backbone in ("gcn", "gat"):
            loops = torch.arange(len(x), device=x.device)
            source = torch.cat((source, loops))
            target = torch.cat((target, loops))
        if self.backbone == "gin":
            aggregated = torch.zeros_like(x).index_add_(0, target, x[source])
            return self.transform((1 + self.epsilon) * x + aggregated)
        if self.backbone == "sage":
            degree = torch.bincount(target, minlength=len(x)).to(x).clamp_min(1)
            aggregated = torch.zeros_like(x).index_add_(0, target, x[source]) / degree[:, None]
            return self.transform(torch.cat((x, aggregated), 1))
        transformed = self.transform(x)
        if self.backbone == "gcn":
            degree = torch.bincount(target, minlength=len(x)).to(x).clamp_min(1)
            weights = (degree[source] * degree[target]).rsqrt()
        else:
            scores = F.leaky_relu((transformed[source] * self.source_attention).sum(1)
                                  + (transformed[target] * self.target_attention).sum(1), 0.2)
            maximum = x.new_full((len(x),), -torch.inf)
            maximum.scatter_reduce_(0, target, scores, reduce="amax", include_self=True)
            exponential = (scores - maximum[target]).exp()
            total = x.new_zeros(len(x)).index_add_(0, target, exponential)
            weights = exponential / total[target].clamp_min(1e-8)
        return torch.zeros_like(transformed).index_add_(0, target, transformed[source] * weights[:, None]) + self.bias


class GraphClassifier(nn.Module):
    def __init__(self, inputs, classes, backbone="gin", hidden=64):
        super().__init__()
        self.layers = nn.ModuleList([Layer(inputs, hidden, backbone),
                                     Layer(hidden, hidden, backbone), Layer(hidden, hidden, backbone)])
        self.normalization = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(3)])
        self.head = nn.Linear(hidden, classes)

    def forward(self, x, edges, graph_ids, graph_count):
        for layer, normalization in zip(self.layers, self.normalization):
            x = F.relu(normalization(layer(x, edges)))
        counts = torch.bincount(graph_ids, minlength=graph_count).to(x).clamp_min(1)
        pooled = x.new_zeros((graph_count, x.shape[1])).index_add_(0, graph_ids, x) / counts[:, None]
        return self.head(F.dropout(pooled, 0.5, self.training))


def pack(graphs, indices, selected, device):
    nodes, edges, graph_ids, labels = [], [], [], []
    offset = 0
    for graph_id, index in enumerate(indices):
        graph = graphs[index]
        x = torch.as_tensor(graph["x"][:, selected], dtype=torch.float32)
        edge = torch.as_tensor(graph["edges"], dtype=torch.long).T
        edge = torch.cat((edge, edge.flip(0)), 1)
        nodes.append(x)
        edges.append(edge + offset)
        graph_ids.append(torch.full((len(x),), graph_id, dtype=torch.long))
        labels.append(graph["y"])
        offset += len(x)
    return (torch.cat(nodes).to(device), torch.cat(edges, 1).to(device),
            torch.cat(graph_ids).to(device), torch.tensor(labels, device=device))
