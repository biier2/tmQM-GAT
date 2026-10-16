# -*- coding: utf-8 -*-
"""扰动归因重跑 — 新 A 变体 9 模型集成 (严格口径: 同协议 RandomState(123) 采样 2000, 逐原子零掩码)
attribution = base_pred − masked_pred (正负归因, 集成均值)
donor = {N,O,S,P,Cl,Br,I}; 第二变体加金属键合碳 (与金属相邻的 C)
输出 perturbation_A_results.json
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, torch, torch.nn as nn

device = torch.device('cpu')
torch.set_num_threads(8)
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']
N_TARGETS = 8
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
    def __init__(self, node_dim=8, edge_dim=6, hidden=HIDDEN, n_layers=N_LAYERS,
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

# ---------- 加载 ----------
DATA = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
N = DATA['y_all'].size(0)
ck = torch.load('ensemble_checkpoint_A_8d6d.pt', map_location='cpu', weights_only=False)
models = []
for mdata in ck['models']:
    m = XYZ_GNN(8, 6); m.eval()
    state = {k.replace('_orig_mod.', ''): v for k, v in mdata['state'].items()}
    m.load_state_dict(state)
    models.append((m, mdata['y_mean'], mdata['y_std']))
print(f'{len(models)} 模型加载完毕', flush=True)

DONOR_Z = {7, 8, 15, 16, 17, 35, 53}

def ensemble_pred(x, ei, ea):
    outs = []
    for m, ym, ys in models:
        with torch.no_grad():
            p = m({'x': x, 'edge_index': ei, 'edge_attr': ea}).squeeze(0)
        outs.append((p * ys + ym).numpy())
    return np.mean(np.stack(outs, 0), axis=0)

# ---------- 采样 (同协议 RandomState(123), n=2000, 全域) ----------
N_PERTURB = 2000
perturb_idx = np.random.RandomState(123).choice(N, N_PERTURB, replace=False)
print(f'采样 {N_PERTURB} 个络合物', flush=True)

donor_frac = {t: [] for t in targets}      # 每络合物 donor |attr| 占比
donor_mc_frac = {t: [] for t in targets}   # donor + 金属键合碳
t0 = time.time()
for ii, i in enumerate(perturb_idx):
    g = {'x': torch.as_tensor(DATA['node8s'][i], dtype=torch.float32),
         'edge_index': DATA['ei_ls'][i],
         'edge_attr': DATA['e6_ls'][i]}
    x = g['x']; ei = g['edge_index']; ea = g['edge_attr']
    n_atoms = x.size(0)
    zs = [int(x[a, 0].item()) for a in range(n_atoms)]
    base = ensemble_pred(x, ei, ea)
    # 金属 + 金属键合碳
    metal_idx = [a for a in range(n_atoms) if x[a, 7] > 0.5]
    mc_carbons = set()
    for a in range(n_atoms):
        if zs[a] != 6: continue
        nbrs = set(ei[1, ei[0] == a].tolist()) | set(ei[0, ei[1] == a].tolist())
        if any(b in metal_idx for b in nbrs):
            mc_carbons.add(a)
    atom_attrs = np.zeros((n_atoms, 8))
    for a in range(n_atoms):
        xm = x.clone()
        xm[a] = 0.0
        masked = ensemble_pred(xm, ei, ea)
        atom_attrs[a] = base - masked
    for p in range(8):
        total_abs = np.abs(atom_attrs[:, p]).sum()
        d_abs = sum(abs(atom_attrs[a, p]) for a in range(n_atoms) if zs[a] in DONOR_Z)
        dm_abs = d_abs + sum(abs(atom_attrs[a, p]) for a in mc_carbons)
        donor_frac[targets[p]].append(d_abs / (total_abs + 1e-8))
        donor_mc_frac[targets[p]].append(dm_abs / (total_abs + 1e-8))
    if (ii + 1) % 200 == 0:
        print(f'  {ii+1}/{N_PERTURB} [{time.time()-t0:.0f}s]', flush=True)

out = {}
for t in targets:
    d = np.array(donor_frac[t]) * 100
    dm = np.array(donor_mc_frac[t]) * 100
    out[t] = {'donor_mean': float(d.mean()), 'donor_std': float(d.std()),
              'donor_mc_mean': float(dm.mean()), 'donor_mc_std': float(dm.std())}
    print(f'{t:14s}: donor {d.mean():.1f}% ± {d.std():.1f}% | +M-C {dm.mean():.1f}% ± {dm.std():.1f}%', flush=True)
json.dump(out, open('perturbation_A_results.json', 'w'), indent=2)
print(f'SAVED perturbation_A_results.json [{time.time()-t0:.0f}s 总]')
