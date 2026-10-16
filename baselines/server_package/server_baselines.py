# -*- coding: utf-8 -*-
"""GCN/GIN/3D-GCN 基线 — 新域 (图同构重建) 重跑
协议与 baseline_gnn.ipynb / build_3dgcn.py 一致: 5×384d, LayerNorm残差, 金属中心池化,
bs=512, torch.compile, 200 ep 早停40, 余弦重启, seed 42 单模型
划分: tr = fold0_train(51,214), va = fold0_val(25,607), te = te_idx(13,556)
输出 baseline_results_new.json
"""
import os, sys, time, json
import numpy as np, torch, torch.nn as nn, torch.optim as optim

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

from sklearn.metrics import r2_score, mean_absolute_error

targets = ["gap","homo","lumo","dipole","metal_charge","E_elec","E_disp","polarizability"]
N_TARGETS = len(targets)
HIDDEN = 384; N_LAYERS = 5; DROPOUT = 0.10

# ---------- 数据 ----------
DATA = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
N = DATA['y_all'].size(0)
y_all = DATA['y_all'].float()
split = np.load('ensemble_split_new.npz', allow_pickle=True) if os.path.exists('ensemble_split_new.npz') else None
idx_all = np.random.RandomState(42).permutation(N)
n_test = int(N * 0.15)
te_idx = idx_all[:n_test]
cv_idx = idx_all[n_test:]
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold0_tr, fold0_va = list(kf.split(cv_idx))[0]
tr_idx = cv_idx[fold0_tr]; va_idx = cv_idx[fold0_va]
print(f'train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}', flush=True)

def build_graphs(node3=False):
    graphs = []
    for i in range(N):
        x = torch.as_tensor(DATA['node8s'][i], dtype=torch.float32)
        if node3:
            x = torch.cat([x, torch.as_tensor(DATA['node3s'][i], dtype=torch.float32)], 1)
        graphs.append({'x': x, 'edge_index': DATA['ei_ls'][i]})
    return graphs

# ---------- 模型 ----------
def scatter_mean_per_graph(h, batch, n):
    out = torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
    cnt = torch.zeros(n, device=h.device, dtype=h.dtype)
    cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)

def scatter_max_per_graph(h, batch, n):
    out = torch.full((n, h.size(1)), float("-inf"), device=h.device, dtype=h.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce="amax", include_self=False)
    return out

class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x, ei):
        src, dst = ei
        Nn = x.size(0)
        msg = x[src]
        aggr = torch.zeros(Nn, x.size(1), device=x.device, dtype=x.dtype)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        deg = torch.zeros(Nn, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst, torch.ones_like(src, dtype=x.dtype))
        deg = deg.clamp(min=1)
        return self.lin(aggr / deg.unsqueeze(-1))

class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
    def forward(self, x, ei):
        src, dst = ei
        Nn = x.size(0)
        aggr = torch.zeros(Nn, x.size(1), device=x.device, dtype=x.dtype)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(x[src]), x[src])
        return self.mlp((1 + self.eps) * x + aggr)

class BaselineGNN(nn.Module):
    def __init__(self, conv_type, node_dim=8):
        super().__init__()
        self.embed = nn.Linear(node_dim, HIDDEN)
        if conv_type == "gin":
            self.convs = nn.ModuleList([GINConv(HIDDEN, HIDDEN) for _ in range(N_LAYERS)])
        else:
            self.convs = nn.ModuleList([GCNConv(HIDDEN, HIDDEN) for _ in range(N_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
        self.drop = nn.Dropout(DROPOUT)
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN*3, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN//2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN//2, N_TARGETS))
    def forward(self, g, batch=None):
        x, ei = g['x'], g['edge_index']
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, ei)); h = self.drop(h)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        mm = (x[:, 7] > 0.5)
        mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

# ---------- 训练 ----------
y_all_gpu = y_all.to(device)
transfer_stream = torch.cuda.Stream()

def train_baseline(graphs, conv_type, node_dim, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    g_x = [g['x'] for g in graphs]
    g_ei = [g['edge_index'] for g in graphs]
    def collate_fast(indices):
        x_list = [g_x[i] for i in indices]; ei_list = [g_ei[i] for i in indices]
        batch_list = []; offset = 0
        for k, x in enumerate(x_list):
            n = x.size(0)
            batch_list.append(torch.full((n,), k, dtype=torch.long))
            ei_list[k] = ei_list[k] + offset; offset += n
        with torch.cuda.stream(transfer_stream):
            batched = {'x': torch.cat(x_list, 0).to(device, non_blocking=True),
                       'edge_index': torch.cat(ei_list, 1).to(device, non_blocking=True)}
            batch_idx = torch.cat(batch_list, 0).to(device, non_blocking=True)
        return batched, batch_idx

    model = BaselineGNN(conv_type, node_dim).to(device)
    try:
        model = torch.compile(model, mode="reduce-overhead"); compiled = True
    except Exception:
        compiled = False
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    scaler = GradScaler("cuda")
    y_tr = y_all_gpu[tr_idx]
    y_mean_local = y_tr.mean(0); y_std_local = y_tr.std(0).clamp(min=1e-8)
    y_norm = (y_all_gpu - y_mean_local) / y_std_local

    bs = 512; best_va = float("inf"); best_state = None; patience = 0
    epochs = 200; early_stop = 40
    n_tr = len(tr_idx); n_va = len(va_idx)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        first_pos = perm[:bs]
        first_bidx = tr_idx[first_pos].tolist()
        next_batch = collate_fast(first_bidx)
        next_by = y_norm[first_bidx]
        for s in range(0, n_tr, bs):
            batched, batch_idx = next_batch
            by = next_by
            ns = s + bs
            if ns < n_tr:
                npos = perm[ns:ns+bs]
                nbidx = tr_idx[npos].tolist()
                next_batch = collate_fast(nbidx)
                next_by = y_norm[nbidx]
            torch.cuda.current_stream().wait_stream(transfer_stream)
            opt.zero_grad()
            with autocast("cuda"):
                loss = nn.MSELoss()(model(batched, batch_idx), by)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = []
            for s in range(0, n_va, bs):
                bidx = va_idx[s:s+bs].tolist()
                batched, batch_idx = collate_fast(bidx)
                with autocast("cuda"):
                    vp.append(model(batched, batch_idx).float())
            va_loss = nn.MSELoss()(torch.cat(vp, 0), y_norm[va_idx])
        if va_loss < best_va:
            best_va = va_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= early_stop:
            print(f'  [{conv_type}] Early stop @ ep {ep+1}', flush=True); break
        if (ep+1) % 30 == 0:
            print(f'  [{conv_type}] Ep {ep+1}: va={va_loss.item():.4f} pat={patience}', flush=True)
    model.load_state_dict(best_state); model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(te_idx), bs):
            bidx = te_idx[s:s+bs].tolist()
            batched, batch_idx = collate_fast(bidx)
            with autocast("cuda"):
                preds.append(model(batched, batch_idx).float().cpu())
    preds = torch.cat(preds, 0) * y_std_local.cpu() + y_mean_local.cpu()
    yt = y_all[te_idx].numpy()
    r2 = [float(r2_score(yt[:, i], preds[:, i].numpy())) for i in range(N_TARGETS)]
    mae = [float(mean_absolute_error(yt[:, i], preds[:, i].numpy())) for i in range(N_TARGETS)]
    return {'r2': r2, 'mae': mae, 'mean_r2': float(np.mean(r2))}

results = {}
graphs_2d = build_graphs(node3=False)
graphs_3d = build_graphs(node3=True)
for ct, gs, nd in [("gcn", graphs_2d, 8), ("gin", graphs_2d, 8), ("gcn3d", graphs_3d, 11)]:
    t0 = time.time()
    print(f'\n=== {ct.upper()} ===', flush=True)
    results[ct] = train_baseline(gs, "gin" if ct == "gin" else "gcn", nd)
    print(f'{ct}: mean R² = {results[ct]["mean_r2"]:.4f} [{ (time.time()-t0)/60:.1f} min]', flush=True)
    json.dump(results, open('baseline_results_new.json', 'w'), indent=2)

print('\nFINAL:')
for ct, r in results.items():
    print(f'{ct:6s} mean R² = {r["mean_r2"]:.4f} | ' + ' '.join(f'{v:.3f}' for v in r['r2']))
print('SAVED baseline_results_new.json')
