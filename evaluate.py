"""
Script for evaluation.
"""
import argparse

from collections import Counter

import math
import networkx as nx
import numpy as np
import pandas as pd

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, required=True, choices=["ride-pooling", "p2p", "regions"]
)
parser.add_argument("--ride-pooling-date", type=str, required=False)
parser.add_argument("--gnutella-version", type=int, default=None)
parser.add_argument("--clusters-csv", type=str, required=True)
args = parser.parse_args()
print(args)

if args.dataset == "regions":
    G = utils.build_regions_graph()
    base_res = utils.calculate_regions_metric()
elif args.dataset == "p2p":
    G = utils.build_gnutella_graph(args.gnutella_version)
    base_res = len(G.edges)
elif args.dataset == "ride-pooling":
    requests, rides = utils.load_csvs(args.ride_pooling_date)
    G = utils.build_shareability_graph(requests, rides)
    base_res = utils.calculate_results(rides, requests)
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

clusters = pd.read_csv(args.clusters_csv, index_col=0)
num_edges = len(G.edges)

to_remove = []
for edge in G.edges:
    if utils.get_cluster(clusters, edge[0]) != utils.get_cluster(clusters, edge[1]):
        to_remove += [edge]

for edge in to_remove:
    G.remove_edge(*edge)

if args.dataset == "regions":
    new_res = utils.calculate_regions_metric(clusters)
elif args.dataset == "p2p":
    new_res = len(G.edges)
elif args.dataset == "ride-pooling":
    to_drop = []
    for i, r in rides.iterrows():
        if len({utils.get_cluster(clusters, idx) for idx in r["indexes"]}) != 1:
            to_drop.append(i)
    rides = rides.drop(index=to_drop)
    new_res = utils.calculate_results(rides, requests)
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

cl_counter = Counter(clusters["cluster"]).items()
cl_mean = np.mean([s for _, s in cl_counter])
cl_std = np.std([s for _, s in cl_counter])
cl_num = len(cl_counter)
experiment_name = args.clusters_csv.split("/")[-1][:-4]

comp_num = 0
comp_size = {}
comp_epidemic_thr = {}
for c in nx.connected_components(G.to_undirected()):
    S = G.subgraph(c).copy()
    d = np.array([v for k, v in dict(S.degree).items()])
    comp_size[comp_num] = len(S)
    comp_epidemic_thr[comp_num] = (
        round(np.mean(d) / np.mean(d**2), 1) if np.mean(d) != 0 else 1
    )
    comp_num += 1

expect_epidemic_thr, N = 0, len(G.nodes)
for c in comp_size:
    expect_epidemic_thr += (float(comp_size[c]) / N) * comp_epidemic_thr[c]

print(
    f"{experiment_name},{cl_num},{base_res},{new_res/base_res:.2f},{cl_mean:.2f},{cl_std:.2f},{comp_num},{expect_epidemic_thr:.2f}"
)
