# -*- coding: utf-8 -*-
"""Shared code for the Fig. 4 zero-padded 11d/23d retrain (review comment id57).

Layout contract (must match fig4_infer.py / variant D):
  x        : (N, 11) = 8 SMILES atom features + 3 BO statistics (node dims 8:11)
  edge_attr: (E, 23) = 6 bond features + 16 RBF (dims 6:22) + 1 WBO (dim 22)

For this retrain ALL geometry-derived channels are identically zero in training,
validation and test:
  x[:, 8:11]      <- 0   (BO neighbour count / sum / max)
  edge_attr[:, 6:]<- 0   (16 RBF distances + 1 WBO)

The arrays are kept at 11/23 columns (not sliced down to 8/6) so the resulting
state_dict is a drop-in for the XYZ_GNN_Eval class in figures/final_render_scripts/fig4_infer.py.
"""
import numpy as np
import torch
import torch.nn as nn

HIDDEN = 384
N_LAYERS = 5
GAT_HEADS = 12
DROPOUT = 0.10
N_TARGETS = 8
NODE_DIM = 11
EDGE_DIM = 23
ZERO_NODE_SLICE = slice(8, 11)
ZERO_EDGE_SLICE = slice(6, 23)


def segment_softmax(scores, dst, n_nodes):
    scores_f32 = scores.float()
    max_v = torch.full((n_nodes, scores_f32.size(1)), float('-inf'),
                       device=scores_f32.device, dtype=torch.float32)
    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)),
                          scores_f32, reduce='amax', include_self=False)
    exp_s = torch.exp(scores_f32 - max_v[dst])
    sum_v = torch.zeros(n_nodes, scores_f32.size(1), device=scores_f32.device,
                        dtype=torch.float32)
    sum_v.scatter_add_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), exp_s)
    return (exp_s / (sum_v[dst] + 1e-8)).to(scores.dtype)


class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim_per_head, edge_dim=0, heads=4):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim_per_head
        total_dim = out_dim_per_head * heads
        self.node_lin = nn.Linear(in_dim, total_dim, bias=False)
        self.attn_src = nn.Linear(total_dim, heads, bias=False)
        self.attn_dst = nn.Linear(total_dim, heads, bias=False)
        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim > 0 else None
        self.act = nn.ReLU()

    def forward(self, x, ei, ea=None):
        src, dst = ei[0], ei[1]
        n_nodes = x.size(0)
        h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0:
            attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, n_nodes)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1)
        msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(n_nodes, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out)


def scatter_mean_per_graph(h, batch, n_graphs):
    out = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
    cnt = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
    cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)


def scatter_max_per_graph(h, batch, n_graphs):
    out = torch.full((n_graphs, h.size(1)), float('-inf'), device=h.device, dtype=h.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h,
                        reduce='amax', include_self=False)
    return out


class XYZ_GNN(nn.Module):
    """Same parameter names as XYZ_GNN_Eval in fig4_infer.py."""

    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden=HIDDEN,
                 n_layers=N_LAYERS, output_dim=N_TARGETS, dropout=DROPOUT,
                 heads=GAT_HEADS):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([
            GATConv(hidden, hidden // heads, edge_dim, heads=heads)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim))

    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, ei, ea))
            h = self.drop(h)
        if batch is None:
            mm = (x[:, 7] > 0.5)
            mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        mm = (x[:, 7] > 0.5)
        mf = (scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any()
              else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype))
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))


def build_zeroed_graph(data, i):
    """One 11d/23d graph with geometry channels forced to zero."""
    n8 = torch.as_tensor(data['node8s'][i], dtype=torch.float32)
    n3 = torch.as_tensor(data['node3s'][i], dtype=torch.float32)
    x = torch.cat([n8, torch.zeros_like(n3)], dim=1)          # (n, 11)
    e6 = torch.as_tensor(data['e6_ls'][i], dtype=torch.float32)
    rbf = torch.as_tensor(data['rbf_ls'][i], dtype=torch.float32)
    bo = torch.as_tensor(data['bo1_ls'][i], dtype=torch.float32)
    ea = torch.cat([e6, torch.zeros_like(rbf), torch.zeros_like(bo)], dim=1)  # (e, 23)
    assert x.size(1) == NODE_DIM and ea.size(1) == EDGE_DIM, (x.shape, ea.shape)
    return {'x': x, 'edge_index': torch.as_tensor(data['ei_ls'][i]).long(), 'edge_attr': ea}


def build_graphs(data, indices):
    """List of zeroed graphs aligned with `indices` (global data indices)."""
    return [build_zeroed_graph(data, int(i)) for i in indices]


def check_layout(graphs, spots=(0, -1)):
    for k in spots:
        g = graphs[k]
        assert g['x'].size(1) == NODE_DIM, g['x'].shape
        assert g['edge_attr'].size(1) == EDGE_DIM, g['edge_attr'].shape
        if g['x'].numel():
            assert float(g['x'][:, ZERO_NODE_SLICE].abs().max()) == 0.0
        if g['edge_attr'].numel():
            assert float(g['edge_attr'][:, ZERO_EDGE_SLICE].abs().max()) == 0.0


def collate_graphs(graph_list):
    x_list, ei_list, ea_list, batch_list = [], [], [], []
    offset = 0
    for k, g in enumerate(graph_list):
        n = g['x'].size(0)
        x_list.append(g['x'])
        ei_list.append(g['edge_index'] + offset)
        ea_list.append(g['edge_attr'])
        batch_list.append(torch.full((n,), k, dtype=torch.long))
        offset += n
    return ({'x': torch.cat(x_list, dim=0),
             'edge_index': torch.cat(ei_list, dim=1),
             'edge_attr': torch.cat(ea_list, dim=0)},
            torch.cat(batch_list, dim=0))


def collate_pairs(pairs):
    """DataLoader collate for [(global_idx, graph), ...]."""
    gidx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    batched, batch_idx = collate_graphs([p[1] for p in pairs])
    return gidx, batched, batch_idx


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, indices):
        self.graphs = graphs
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        return int(self.indices[k]), self.graphs[k]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
