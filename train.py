"""
Script for training.
"""

import argparse

from collections import Counter

import pandas as pd
import torch
import networkx as nx
import numpy as np
from torch_geometric import utils as geo_utils
import utils
from model import GNN


parser = argparse.ArgumentParser(description="Training code for clustering methods")
parser.add_argument(
    "--dataset", type=str, required=True, choices=["ride-pooling", "p2p", "regions"], help="Dataset type"
)
parser.add_argument("--ride-pooling-date", type=str, required=False, help="Date for ride pooling network")
parser.add_argument("--gnutella-version", type=int, default=None, help="Version of Gnutella p2p network")
parser.add_argument("--output-csv", type=str, required=True, help="Output save csv path")
parser.add_argument("--delta", type=float, default=0.85, help="Delta for laplacian matrix normalization")
parser.add_argument("--seed", type=int, default=123, help="Random seed")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
parser.add_argument("--num-clusters", type=int, default=3, help="Number of clusters")
parser.add_argument("--mp-units", type=str, default="[16, 16]", help="Sizes of Message Passing layers")
parser.add_argument("--mlp-units", type=str, default="[3]", help="Sizes of MLP layers")
parser.add_argument("--dmon", action="store_true", help="Enable DMON")
parser.add_argument(
    "--kappa",
    type=str,
    default="[0.,0.,0.,0.,0.,1.,1.,0,0,0]",
    help="Weights for loss that allow training various clustering network within one framework",
)
parser.add_argument("--laplacian", action="store_true", help="Enable laplacian normalization")
parser.add_argument("--edge-weight", type=str, choices=["u", "frac_u"], help="Edge weight to use")
args = parser.parse_args()
print(args)

mp_units = utils.str_to_list(args.mp_units)
mlp_units = utils.str_to_list(args.mlp_units)
kappa = utils.str_to_list(args.kappa, dtype=float)

# Load the network data
assert len(kappa) == 8
if args.dataset == "regions":
    G = utils.build_regions_graph()
elif args.dataset == "p2p":
    G = utils.build_gnutella_graph(args.gnutella_version)
elif args.dataset == "ride-pooling":
    requests, rides = utils.load_csvs(args.ride_pooling_date)
    G = utils.build_shareability_graph(requests, rides)
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

# Find components
H = G.copy()
negative_edges = list(filter(lambda e: e[2] <= 0, (e for e in G.edges.data(args.edge_weight))))
le_ids = list(e[:2] for e in negative_edges)
H.remove_edges_from(le_ids)
comps = nx.connected_components(H)
components = np.zeros(len(G.nodes), dtype=int)
for i, c in enumerate(comps):
    for e in c:
        components[e] = i
del H, le_ids, negative_edges, comps
unique, counts = np.unique(components, return_counts=True)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# Setup node features
if args.dataset == "p2p":
    x = torch.ones(len(G.nodes), 32).to(device)
else:
    x = torch.eye(len(G.nodes)).to(device)

delta = args.delta
num_clusters = args.num_clusters

adj_matrix = nx.adjacency_matrix(G, weight=args.edge_weight)
edge_weight = torch.empty(len(G.edges))
edge_index = torch.empty((2, len(G.edges)), dtype=torch.int64)

for i, e in enumerate(G.edges):
    edge_index[0, i] = e[0]
    edge_index[1, i] = e[1]
    edge_weight[i] = max(adj_matrix[e], 0) if args.laplacian else adj_matrix[e]

del adj_matrix

# Normalization
if args.laplacian:
    edge_index, edge_weight = geo_utils.get_laplacian(edge_index, edge_weight, normalization="sym")
    L = geo_utils.to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    adj = torch.eye(L.shape[-1]) - delta * L
else:
    adj = geo_utils.to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    for i in range(adj.shape[0]):
        adj[i, i] = 1 - delta

edge_index, edge_weight = geo_utils.dense_to_sparse(adj)
edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
del adj

# Setup the Network
model = GNN(
    mp_units,
    "ReLU",
    x.shape[1],
    num_clusters,
    mlp_units,
    "ReLU",
    args.dmon,
    kappa,
)

model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
x, edge_index, edge_weight = x.to(device), edge_index.to(device), edge_weight.to(device)


def train(model, optimizer, x, edge_index, edge_weight, components, unique, counts, kappa):
    """
    Training iteration
    """
    model.train()
    optimizer.zero_grad()
    clusters, loss = model(x, edge_index, edge_weight, components, unique, counts)
    tot_loss = torch.matmul(torch.stack(list(loss.values())), torch.tensor(kappa[: len(loss)]).to(device))
    tot_loss.backward()
    loss["total"] = tot_loss
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, x, edge_index, edge_weight, components, unique, counts):
    """
    Evaluation iteration
    """
    model.eval()
    softmax, _ = model(x, edge_index, edge_weight, components, unique, counts)
    return softmax.argmax(axis=1).cpu(), softmax.cpu()


# Training loop
for epoch in range(1, args.epochs + 1):
    train_loss = train(model, optimizer, x, edge_index, edge_weight, components, unique, counts, kappa)
    if epoch % 10 == 0:
        clusters, _ = test(model, x, edge_index, edge_weight, components, unique, counts)
        clusters = dict(Counter(clusters.numpy()))
        train_loss = {k: round(v.item(), 3) for i, (k, v) in enumerate(train_loss.items())}
        print(f"Epoch: {epoch:03d}, Loss: {train_loss['total']} | {clusters}")
        print({k: v for i, (k, v) in enumerate(train_loss.items()) if k != "total" and kappa[i] != 0})

clusters, softmax = test(model, x, edge_index, edge_weight, components, unique, counts)
out = {"cluster": clusters}
for c in range(args.num_clusters):
    out[f"soft_{c}"] = softmax[:, c].cpu()
pd.DataFrame(out).to_csv(args.output_csv)
