"""
Utility methods.
"""

from typing import Dict, List, TypeVar, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
import platform
import pulp
import torch


def str_to_list(s: str, dtype: TypeVar = int) -> List:
    s = s.replace("[", "").replace("]", "")
    if not s:
        return []
    return [dtype(t) for t in s.split(",")]


def build_regions_graph() -> nx.Graph:
    """
    Builds regions graph from graphml data
    """
    G = nx.read_graphml("data/regions/graph.graphml")
    remove = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(remove)
    nodes = G.nodes(data=True)
    nodes_dict = {n: idx for n, idx in zip(G.nodes, range(len(nodes)))}
    H = nx.Graph()

    for i, j, data in G.edges(data=True):
        d = data["weight"]
        if ("CITIZENS" not in nodes[i]) or ("CITIZENS" not in nodes[j]):
            continue
        a_i = float(nodes[i]["CITIZENS"])
        a_j = float(nodes[j]["CITIZENS"])
        H.add_edge(nodes_dict[i], nodes_dict[j], u=((a_i * a_j) / d))
    new_weights = {}
    for i, j, data in H.edges(data=True):
        new_weights[(i, j)] = data["u"]
        new_weights[(j, i)] = data["u"]
    maxes = {}
    for i in H.nodes:
        maxes[i] = max([v for k, v in new_weights.items() if i in k])
    for i, j, data in H.edges(data=True):
        data["u"] = 0.5 * ((data["u"] / maxes[i]) + (data["u"] / maxes[j]))
    return H


def build_gnutella_graph(version: int) -> nx.Graph:
    """
    Builds gnutella p2p graph from txt data for version `version`
    """
    file = f"data/p2p/p2p-Gnutella{version:02d}.txt"
    G = nx.Graph()
    with open(file, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i < 4:
                continue
            a, b = line.split("\t")
            a, b = int(a), int(b)
            G.add_edge(a, b, u=1)
    G = nx.relabel_nodes(G, {v: i for i, v in enumerate(G.nodes)})
    degrees = G.degree
    for u, v, d in G.edges(data=True):
        d["u"] = ((1.0 / degrees[u]) + (1.0 / degrees[v])) / 2.0
    return G


def load_csvs(ride_pooling_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads ride pooling data for a given date
    """
    requests_csv = f"data/ride-pooling/{ride_pooling_date}/requests.csv"
    rides_csv = f"data/ride-pooling/{ride_pooling_date}/rides.csv"
    requests = pd.read_csv(requests_csv, index_col=0)
    rides = pd.read_csv(rides_csv, index_col=0)
    rides["indexes"] = rides.apply(lambda x: str_to_list(x["indexes"]), axis=1)
    return requests, rides


def build_shareability_graph(requests: pd.DataFrame, rides: pd.DataFrame) -> nx.Graph:
    """
    Builds shareability graph from requests and rides csv files
    """
    G = nx.Graph()
    G.add_nodes_from(requests.index)
    edges = []
    _rides = rides.copy()
    times = {}
    for _, row in rides.iterrows():
        idx = row["indexes"]
        if len(idx) < 2:
            times[str(idx)] = row["u_veh"]
        if len(idx) == 2:
            rev = idx[::-1]
            if (str(idx) not in times) and (str(rev) not in times):
                times[str(idx)] = row["u_veh"]
            if str(rev) in times:
                times[str(rev)] = max(times[str(rev)], row["u_veh"])

    for _, row in _rides.iterrows():
        if len(row.indexes) == 2:
            e = row.indexes
            if str(e) not in times:
                continue
            a, b, c = [e[0]], [e[1]], list(e)
            a = times[str(a)]
            b = times[str(b)]
            c = times[str(c)]
            edges.append(
                (
                    row.indexes[0],
                    row.indexes[1],
                    {
                        "u": a + b - c,
                        "frac_u": (a + b - c) / (a + b),
                    },
                )
            )
    G.add_edges_from(edges)
    return G


def match(
    rides: pd.DataFrame,
    requests: pd.DataFrame,
    matching_obj: str = "u_veh",
) -> Dict:
    """
    Calculates optimal matching of rides based on requests
    """
    request_indexes = {}
    request_indexes_inv = {}
    for i, index in enumerate(requests.index.values):
        request_indexes[index] = i
        request_indexes_inv[i] = index

    im_indexes = {}
    im_indexes_inv = {}
    for i, index in enumerate(rides.index.values):
        im_indexes[index] = i
        im_indexes_inv[i] = index

    nR = requests.shape[0]

    def add_binary_row(requests):
        ret = np.zeros(nR)
        for i in requests.indexes:
            ret[request_indexes[i]] = 1
        return ret

    rides["row"] = rides.apply(add_binary_row, axis=1)  # row to be used as constrain in optimization
    m = np.vstack(rides["row"].values).T  # creates a numpy array for the constrains

    rides["index"] = rides.index.copy()

    rides = rides.reset_index(drop=True)

    # optimization
    prob = pulp.LpProblem("Matchingproblem", pulp.LpMinimize)  # problem

    variables = pulp.LpVariable.dicts("r", (i for i in rides.index), cat="Binary")  # decision variables

    cost_col = matching_obj
    if cost_col == "degree":
        costs = rides.indexes.apply(lambda x: -(10 ** len(x)))
    elif cost_col == "u_pax":
        costs = rides[cost_col]  # set the costs
    else:
        costs = rides[cost_col]  # set the costs

    prob += (
        pulp.lpSum([variables[i] * costs[i] for i in variables]),
        "ObjectiveFun",
    )  # ffef

    j = 0  # adding constrains
    for imr in m:
        j += 1
        prob += pulp.lpSum([imr[i] * variables[i] for i in variables if imr[i] > 0]) == 1, "c" + str(j)

    solver = pulp.get_solver(solver_for_pulp())
    solver.msg = False
    prob.solve(solver)

    assert pulp.value(prob.objective) <= sum(costs[:nR]) + 2  # we did not go above original

    locs = {}
    for variable in prob.variables():
        i = int(variable.name.split("_")[1])

        locs[im_indexes_inv[i]] = int(variable.varValue)

    return locs


def solver_for_pulp() -> str:
    system = platform.system()
    if system == "Windows":
        return "GLPK_CMD"
    else:
        return "PULP_CBC_CMD"


def calculate_results(rides: pd.DataFrame, requests: pd.DataFrame) -> float:
    """
    Evaluates optimal matching of rides based on requests
    """
    match(rides, requests)
    fin = rides.loc[rides["selected"] == 1]
    return sum(fin["PassSecTrav_ns"]) - sum(fin["u_veh"])


EPS = 1e-15


def just_balance_pool(
    x: torch.Tensor, adj: torch.Tensor, s: torch.Tensor, mask: torch.Tensor = None, normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Applies the Just Balance pooling operator from the
    `"Simplifying Clustering with Graph Neural Networks" <https://arxiv.org/abs/2207.08779>`_ paper.

    This operator performs a soft clustering of node features and aggregates the graph structure
    accordingly. Implementation adapted from the original GitHub repository:
    https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks

    Args:
        x (torch.Tensor): Node feature matrix of shape [N, F], where N is the number of nodes and F is the number of features.
        adj (torch.Tensor): Adjacency matrix of shape [N, N], typically sparse or dense float tensor.
        s (torch.Tensor): Soft assignment matrix of shape [N, K], where K is the number of clusters.
        mask (torch.Tensor, optional): Mask tensor of shape [N] indicating which nodes are valid (useful for batching).
        normalize (bool, optional): If True, normalizes the assignment matrix `s`. Default is True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Node feature matrix of shape
            - Adjacency matrix.
            - Loss term
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Loss
    ss = torch.matmul(s.transpose(1, 2), s)
    ss_sqrt = torch.sqrt(ss + EPS)
    loss = torch.mean(-_rank3_trace(ss_sqrt))
    if normalize:
        loss = loss / torch.sqrt(torch.tensor(num_nodes * k))

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, loss


def _rank3_trace(x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ijj->i", x)


def get_cluster(clusters: pd.DataFrame, idx: int) -> int:
    return clusters.iloc[idx].cluster


def calculate_regions_metric(clusters: Optional[pd.DataFrame] = None):
    """
    Calculates evaluation metric for the regions network
    """
    res = 0.0
    H = nx.read_graphml("data/regions/graph.graphml")
    dist = pd.read_csv("data/regions/dist_matrix.csv", index_col=0)
    nodes = H.nodes(data=True)
    nodes_dict = {int(n): idx for n, idx in zip(H.nodes, range(len(nodes)))}
    for i in H.nodes:
        for j in H.nodes:
            if i <= j:
                continue
            if ("CITIZENS" not in nodes[i]) or ("CITIZENS" not in nodes[j]):
                continue
            if clusters is not None:
                if get_cluster(clusters, nodes_dict[int(i)]) != get_cluster(clusters, nodes_dict[int(j)]):
                    continue
            d = dist[str(i)][int(j)]
            a_i = float(nodes[i]["CITIZENS"])
            a_j = float(nodes[j]["CITIZENS"])
            res += (a_i * a_j) / d
    return res
