# -*- coding: utf-8 -*-
"""本地 CPU: B/D 变体在 ETKDG 测试图上的 9 模型推断 (补 geom_preds npz + Spearman)
CPU 约 30-60 分钟, 后台运行"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

device = torch.device('cpu')
print(f'Device: {device}', flush=True)
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']
N_TARGETS = len(targets)
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10

def segment_softmax(scores, dst, Nn):
    scores_f32 = scores.float()
    max_v = torch.full((Nn, scores_f32.size(1)), float('-inf'), device=scores_f32.device, dtype=torch.float32)
    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), scores_f32, reduce='amax', include_self=False)
    exp_s = torch.exp(scores_f32 - max_v[dst])
    sum_v = torch.zeros(Nn, scores_f32.size(1), device=scores_f32.device, dtype=torch.float32)
    sum_v.scatter_add_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), exp_s)
    return (exp_s / (sum_v[dst] + 1e-8)).to(scores.dtype)

class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim_per_head, edge_dim=0, heads=4):
        super().__init__(); self.heads, self.out_dim = heads, out_dim_per_head
        total_dim = out_dim_per_head * heads
        self.node_lin = nn.Linear(in_dim, total_dim, bias=False)
        self.attn_src = nn.Linear(total_dim, heads, bias=False)
        self.attn_dst = nn.Linear(total_dim, heads, bias=False)
        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim > 0 else None
        self.act = nn.ReLU()
    def forward(self, x, ei, ea=None):
        src, dst = ei[0], ei[1]; Nn = x.size(0); h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0: attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, Nn)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(Nn, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
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
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce='amax', include_self=False)
    return out

class XYZ_GNN(nn.Module):
    def __init__(self, node_dim=11, edge_dim=23, hidden=HIDDEN, n_layers=N_LAYERS,
                 output_dim=N_TARGETS, dropout=DROPOUT, heads=GAT_HEADS):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            mm = (x[:, 7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        mm = (x[:, 7] > 0.5)
        mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

DATA = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
te_idx = DATA['te_idx']
y_all = DATA['y_all']
ETKDG = torch.load('etkdg_test_graphs.pt', map_location='cpu', weights_only=False)
ok = ETKDG['ok']
n_te_trunc = int(0.15 * DATA['y_all'].size(0))
idxs = np.array([i for i in range(n_te_trunc) if ok[i]])
print(f'ETKDG 成功子集: {len(idxs)}', flush=True)

for VNAME, nd, ed in [('B_8d22d', 8, 22), ('D_11d23d', 11, 23)]:
    ck = torch.load(f'ensemble_checkpoint_{VNAME}.pt', map_location='cpu', weights_only=False)
    models = ck['models']
    gs = [ETKDG[f'graphs_{VNAME[0]}'][i] for i in idxs]
    all_preds = []
    for mi, mdata in enumerate(models):
        model = XYZ_GNN(node_dim=nd, edge_dim=ed)
        state = {k.replace('_orig_mod.', ''): v for k, v in mdata['state'].items()}
        model.load_state_dict(state); model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(gs), 128):
                batch_g = gs[s:s+128]
                x_list, ei_list, ea_list, bl = [], [], [], []
                off = 0
                for k, g in enumerate(batch_g):
                    n = g['x'].size(0)
                    x_list.append(g['x']); ei_list.append(g['edge_index'] + off)
                    ea_list.append(g['edge_attr']); bl.append(torch.full((n,), k, dtype=torch.long))
                    off += n
                p = model({'x': torch.cat(x_list, 0), 'edge_index': torch.cat(ei_list, 1),
                           'edge_attr': torch.cat(ea_list, 0)}, torch.cat(bl, 0))
                preds.append(p)
        preds = torch.cat(preds, 0)
        all_preds.append((preds * mdata['y_std'] + mdata['y_mean']).numpy())
        print(f'  {VNAME} model {mi+1}/9 完成', flush=True)
    ap = np.stack(all_preds, 0)
    ens = ap.mean(0)
    yt = y_all[te_idx].numpy()[idxs]
    np.savez(f'geom_preds_{VNAME}_etkdg.npz', preds=ap, ens_mean=ens, y_true=yt, idxs=idxs)
    print(f'{VNAME} saved, mean R² = {np.mean([r2_score(yt[:, p], ens[:, p]) for p in range(8)]):.4f}', flush=True)

# Spearman 双几何 (B)
zB = np.load('test_preds_B_8d22d.npz', allow_pickle=False)
eB = np.load('geom_preds_B_8d22d_etkdg.npz', allow_pickle=False)
idB = eB['idxs']
spear = {}
for p in range(8):
    rho, _ = stats.spearmanr(zB['ens_mean'][idB, p], eB['ens_mean'][:, p])
    spear[targets[p]] = float(rho)
print('Spearman (B, XYZ vs ETKDG):', {k: round(v, 4) for k, v in spear.items()}, flush=True)
json.dump({'spearman_xyz_vs_etkdg': spear}, open('geom_spearman.json', 'w'), indent=2)
print('SAVED geom_spearman.json')
