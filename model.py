"""
Implementation of GNN model.
"""
import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric import utils as geo_utils
from torch_geometric.nn import DMoNPooling, GCNConv, Sequential, dense_mincut_pool

import numpy as np
import utils


class GNN(torch.nn.Module):
    """
    Based on https://github.com/FilippoMB/Simplifying-Clustering-with-Graph-Neural-Networks.
    """

    def __init__(
        self,
        mp_units,
        mp_act,
        in_channels,
        n_clusters,
        mlp_units,
        mlp_act,
        dmon,
        kappa,
    ):
        super().__init__()

        self.kappa = kappa
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        if len(mp_units) > 0:
            header = "x, edge_index, edge_weight -> x"
            gcn_layer = GCNConv(in_channels, mp_units[0], normalize=False)
            mp = [(gcn_layer, header), mp_act]
            for i in range(len(mp_units) - 1):
                gcn_layer = GCNConv(mp_units[i], mp_units[i + 1], normalize=False)
                mp.append((gcn_layer, header))
                if (i < len(mp_units) - 2) or (len(mlp_units) > 0) or dmon:
                    mp.append(mp_act)
            self.mp = Sequential("x, edge_index, edge_weight", mp)
            out_chan = mp_units[-1]
        else:
            self.mp = torch.nn.Sequential()
            out_chan = in_channels

        self.mlp = torch.nn.Sequential()
        for i, units in enumerate(mlp_units):
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            if i < len(mlp_units) - 1:
                self.mlp.append(mlp_act)
            out_chan = units

        if dmon:
            self.dmon = DMoNPooling(out_chan, n_clusters)
            out_chan = n_clusters
        else:
            self.dmon = None
        assert out_chan == n_clusters


    def forward(self, x, edge_index, edge_weight, components, unique, counts):
        device = x.device
        x = self.mp(x, edge_index, edge_weight)
        s = self.mlp(x)
        adj = geo_utils.to_dense_adj(
                edge_index, edge_attr=edge_weight, max_num_nodes=x.shape[0]
            )[0]
        adj = torch.nn.functional.relu(adj, inplace=True)

        # DMoN losses
        if self.dmon is None:
            soft = torch.softmax(s, dim=-1)
            spectral_loss, ortho_loss, cluster_loss = (
                torch.zeros([], device=device),
                torch.zeros([], device=device),
                torch.zeros([], device=device),
            )
        else:
            soft, _, _, spectral_loss, ortho_loss, cluster_loss = self.dmon(
                x.unsqueeze(0).to(device), adj.unsqueeze(0).to(device)
            )
            soft, adj = soft.cpu(), adj.cpu()
            soft = soft[0]

        # Balance pooling loss from Simplifying clustering...
        if self.kappa[3] != 0:
            _, _, balance_pool_loss = utils.just_balance_pool(x, adj.to(device), s)
        else:
            balance_pool_loss = torch.zeros([], device=device)

        if self.kappa[4] != 0:
            adj = adj.cpu()
            soft = soft.to(device)
            eye = torch.eye(adj.shape[0]).float()  # .to(device)
            adj_wth_loops = (adj > 0).float() - eye
            sas = (
                torch.mm(torch.mm(soft.T, adj_wth_loops.to(device)).T, soft.T).cpu()
                + eye
            )

            epidemic_thr_loss = []
            N = adj.shape[0]
            for c, count in zip(unique, counts):
                if count == 1:
                    continue
                sas_sub = sas[components == c][:, components == c]
                up = torch.trace(sas_sub)
                down = (torch.diagonal(sas_sub) ** 2).sum()
                thr = (count / N) * (up / down)
                epidemic_thr_loss += [thr]
            epidemic_thr_loss = -torch.tensor(epidemic_thr_loss).sum().to(device)
        else:
            epidemic_thr_loss = torch.zeros([], device=device)

        if self.kappa[5] != 0 or self.kappa[6] != 0:
            x, adj, s = x.cpu(), adj.cpu(), s.cpu()
            _, _, mincut_mc_loss, mincut_orth_loss = dense_mincut_pool(x, adj, s)
            mincut_mc_loss, mincut_orth_loss = mincut_mc_loss.to(
                device
            ), mincut_orth_loss.to(device)
        else:
            mincut_mc_loss, mincut_orth_loss = torch.zeros(
                [], device=device
            ), torch.zeros([], device=device)

        if self.kappa[7] != 0:
            adj = adj.cpu()
            neg_adj = adj.clone()
            sim = (-torch.mm(soft, soft.T) + 1).cpu()
            our_sim_loss = (neg_adj * sim).to(device).sum(dim=-1).mean()
        else:
            our_sim_loss = torch.zeros([], device=device)

        loss = {
            "spectral": spectral_loss,
            "ortho": ortho_loss,
            "cluster": cluster_loss,
            "balance_pool": balance_pool_loss,
            "epidemic_thr": epidemic_thr_loss,
            "mincut_mc": mincut_mc_loss,
            "mincut_orth": mincut_orth_loss,
            "our_sim": our_sim_loss,
        }

        return soft, loss
