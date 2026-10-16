# -*- coding: utf-8 -*-
"""意见2(b): ETKDG+MMFF 部署几何上的测试集推断 (变体 B=8d/22d, D=11d/23d)
官方 XYZ 几何结果已在 server_train_variants.py 的标准测试评估中得出; 本脚本只跑 ETKDG 通道。
输出: geom_preds_{V}.npz + geom_results.json (双几何配对差异)
"""
import os, sys, json
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
from torch.amp import autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
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

# ================= 数据 =================
DATA = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
te_idx = DATA['te_idx']
y_all = DATA['y_all']
ETKDG = torch.load('etkdg_test_graphs.pt', map_location='cpu', weights_only=False)
ok = ETKDG['ok']
print(f'ETKDG 成功: {ok.sum()}/{len(ok)}')

def infer_variant(VNAME, node_dim, edge_dim, graphs_etkdg):
    ck = torch.load(f'ensemble_checkpoint_{VNAME}.pt', map_location='cpu', weights_only=False)
    models = ck['models']
    print(f'{VNAME}: {len(models)} models', flush=True)
    # 预建 index: 按 ok 过滤; 并与训练脚本截断口径 int(0.15*N)=13,556 对齐
    n_te_trunc = int(0.15 * DATA['y_all'].size(0))
    idxs = [i for i in range(len(te_idx)) if ok[i] and i < n_te_trunc]
    gs = [graphs_etkdg[i] for i in idxs]
    all_preds = []
    for mdata in models:
        model = XYZ_GNN(node_dim=node_dim, edge_dim=edge_dim).to(device)
        state = {k.replace('_orig_mod.', ''): v for k, v in mdata['state'].items()}
        model.load_state_dict(state); model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(gs), 64):
                batch_g = gs[s:s+64]
                x_list2, ei_list2, ea_list, bl = [], [], [], []
                off = 0
                for k, g in enumerate(batch_g):
                    n = g['x'].size(0)
                    x_list2.append(g['x']); ei_list2.append(g['edge_index'] + off)
                    ea_list.append(g['edge_attr']); bl.append(torch.full((n,), k, dtype=torch.long))
                    off += n
                batched = {'x': torch.cat(x_list2, 0).to(device),
                           'edge_index': torch.cat(ei_list2, 1).to(device),
                           'edge_attr': torch.cat(ea_list, 0).to(device)}
                batch_idx = torch.cat(bl, 0).to(device)
                with autocast('cuda'):
                    p = model(batched, batch_idx).float().cpu()
                preds.append(p)
        preds = torch.cat(preds, 0)
        all_preds.append((preds * mdata['y_std'] + mdata['y_mean']).numpy())
        del model; torch.cuda.empty_cache()
        print(f'  model done ({len(all_preds)})', flush=True)
    return np.stack(all_preds, 0), idxs

out = {}
for VNAME, nd, ed in [('B_8d22d', 8, 22), ('D_11d23d', 11, 23)]:
    ap, idxs = infer_variant(VNAME, nd, ed, ETKDG[f'graphs_{VNAME[0]}'])
    ens = ap.mean(0)
    yt = y_all[te_idx].numpy()[idxs]
    res = {}
    for i, prop in enumerate(targets):
        res[prop] = {'ens_r2': float(r2_score(yt[:, i], ens[:, i])),
                     'ens_mae': float(mean_absolute_error(yt[:, i], ens[:, i]))}
    res['mean_ens_r2'] = float(np.mean([res[p]['ens_r2'] for p in targets]))
    out[VNAME] = {'etkdg': res}
    np.savez(f'geom_preds_{VNAME}_etkdg.npz', preds=ap, ens_mean=ens, y_true=yt, idxs=np.array(idxs))
    print(f'{VNAME} ETKDG mean ensemble R2 = {res["mean_ens_r2"]:.4f}', flush=True)

# 与官方 XYZ 几何结果 (test_preds_{V}.npz) 配对比较
for VNAME in ['B_8d22d', 'D_11d23d']:
    z = np.load(f'test_preds_{VNAME}.npz')
    ens_xyz = z['ens_mean']; yt_xyz = z['y_true']
    e = np.load(f'geom_preds_{VNAME}_etkdg.npz')
    idxs = e['idxs']
    diff = {}
    for i, prop in enumerate(targets):
        y_s = yt_xyz[idxs, i]
        r2_xyz = r2_score(y_s, ens_xyz[idxs, i]); r2_etkdg = r2_score(y_s, e['ens_mean'][:, i])
        diff[prop] = {'r2_xyz': float(r2_xyz), 'r2_etkdg': float(r2_etkdg),
                      'dr2': float(r2_etkdg - r2_xyz),
                      'mae_xyz': float(mean_absolute_error(y_s, ens_xyz[idxs, i])),
                      'mae_etkdg': float(mean_absolute_error(y_s, e['ens_mean'][:, i]))}
    out[VNAME]['paired_xyz_vs_etkdg'] = diff
    print(json.dumps(diff, indent=2)[:900])

with open('geom_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print('SAVED geom_results.json')
