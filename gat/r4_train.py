# -*- coding: utf-8 -*-
"""R4: 跨泛函 Mayer BO 对照实验 — 训练 (服务器 GPU)
数据: r4_graphs.pt (74k tmQMg 图, ωB97M-V 标签 7 性质)
模型A: SMILES GAT (node 8d / edge 6d)  模型B: SMILES + TPSSh BO 增广 (node 11d / edge 6d)
协议: 论文同款 (12头×32d×5层, readout 1152->384->192->7, AdamW lr=1e-3, AMP,
      BATCH=128, EPOCHS=150, EARLY_STOP=40, 划分 RandomState(42) 70/15/15, 3 seeds)
性能: numpy 向量化 collate + DataLoader 多进程 (NW=8) + pin_memory + 异步传输
输出: r4_results.json — per-config per-seed per-prop R²/MAE
"""
import json, os, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

PROPS = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10
BATCH = int(os.environ.get('R4_BATCH', '128'))
EPOCHS = int(os.environ.get('R4_EPOCHS', '150'))
EARLY_STOP = int(os.environ.get('R4_EARLY_STOP', '40'))
N_OUT = 7
SEEDS = [int(x) for x in os.environ.get('R4_SEEDS', '42,123,456').split(',')]
SMOKE = int(os.environ.get('R4_SMOKE', '0'))
NW = int(os.environ.get('R4_NW', '8'))      # DataLoader 进程数 (16 核给 8)

# ---------- 数据 ----------
data = torch.load('r4_graphs.pt', weights_only=False)
graphs = [{'x': torch.tensor(g['x11'], dtype=torch.float32),
           'edge_index': torch.tensor(g['edge_index'], dtype=torch.long),
           'edge_attr': torch.tensor(g['edge_attr'], dtype=torch.float32)}
          for g in data['graphs']]
y_all = data['y']
if SMOKE:
    graphs = graphs[:SMOKE]
    y_all = y_all[:SMOKE]
    print(f'[SMOKE] 子集 {SMOKE} 图', flush=True)
print(f'数据: {len(graphs)} 图, 标签 {y_all.shape}, NW={NW}', flush=True)

idx = np.random.RandomState(42).permutation(len(graphs))
n_tr = int(len(graphs) * 0.7); n_va = int(len(graphs) * 0.15)
tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
print(f'划分: train {n_tr} / val {n_va} / test {len(te_idx)}', flush=True)

y_mean = y_all.mean(0); y_std = y_all.std(0).clamp(min=1e-8)
y_norm = (y_all - y_mean) / y_std

# ---------- 数据管道: numpy 向量化 collate + 多进程 ----------
class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, i):
        g = self.graphs[i]
        return g['x'], g['edge_index'], g['edge_attr'], i

def collate_fn(batch):
    ns = np.array([b[0].shape[0] for b in batch], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(ns)[:-1]])
    x_cat = torch.cat([b[0] for b in batch], dim=0)
    ei_cat = torch.cat([b[1] + offsets[k] for k, b in enumerate(batch)], dim=1)
    ea_cat = torch.cat([b[2] for b in batch], dim=0)
    b_cat = torch.from_numpy(np.repeat(np.arange(len(batch), dtype=np.int64), ns))
    gidx = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return {'x': x_cat, 'edge_index': ei_cat, 'edge_attr': ea_cat}, b_cat, gidx

def make_loader(indices, shuffle):
    kw = dict(batch_size=BATCH, shuffle=shuffle, collate_fn=collate_fn,
              num_workers=NW, pin_memory=True, drop_last=False)
    if NW > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(Subset(GraphDataset(graphs), indices), **kw)

# ---------- 模型 (论文 AblationXYZ 协议, 输出 7 维) ----------
def segment_softmax(scores, dst, N):
    scores_f32 = scores.float()
    max_v = torch.full((N, scores_f32.size(1)), float('-inf'), device=scores_f32.device, dtype=torch.float32)
    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), scores_f32, reduce='amax', include_self=False)
    exp_s = torch.exp(scores_f32 - max_v[dst])
    sum_v = torch.zeros(N, scores_f32.size(1), device=scores_f32.device, dtype=torch.float32)
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
        src, dst = ei[0], ei[1]; N = x.size(0); h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0:
            attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, N)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(N, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out)

class R4GAT(nn.Module):
    def __init__(self, node_dim=8, edge_dim=6):
        super().__init__()
        self.node_dim, self.edge_dim = node_dim, edge_dim
        self.embed = nn.Linear(node_dim, HIDDEN)
        self.convs = nn.ModuleList([GATConv(HIDDEN, HIDDEN // GAT_HEADS, edge_dim, heads=GAT_HEADS) for _ in range(N_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
        self.drop = nn.Dropout(DROPOUT)
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN * 3, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN // 2, N_OUT))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        if self.node_dim < x.size(1): x = x[:, :self.node_dim]
        if ea is not None and ea.size(0) > 0 and self.edge_dim < ea.size(1):
            ea = ea[:, :self.edge_dim]
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            mm = (x[:, 7] > 0.5); mf = h[mm].mean(0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        out = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
        cnt = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
        cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
        mean_h = out / cnt.clamp(min=1).unsqueeze(-1)
        max_h = torch.full((n_graphs, h.size(1)), float('-inf'), device=h.device, dtype=h.dtype)
        max_h.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce='amax', include_self=False)
        mm = (x[:, 7] > 0.5)
        if mm.any():
            mout = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
            mout.scatter_add_(0, batch[mm].unsqueeze(-1).expand_as(h[mm]), h[mm])
            mcnt = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
            mcnt.scatter_add_(0, batch[mm], torch.ones(int(mm.sum()), device=h.device, dtype=h.dtype))
            mf = mout / mcnt.clamp(min=1).unsqueeze(-1)
        else:
            mf = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

def to_gpu(batched, batch_idx, gidx):
    return ({k: v.to(device, non_blocking=True) for k, v in batched.items()},
            batch_idx.to(device, non_blocking=True),
            gidx.to(device, non_blocking=True))

def train_eval(node_dim, edge_dim, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = R4GAT(node_dim, edge_dim).to(device)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    crit = nn.MSELoss(reduction='none')
    scaler = GradScaler('cuda')
    best_va, best_state, patience = float('inf'), None, 0
    yn = y_norm.to(device)
    tr_loader = make_loader(tr_idx, shuffle=True)
    va_loader = make_loader(va_idx, shuffle=False)
    for ep in range(EPOCHS):
        m.train()
        for batched, batch_idx, gidx in tr_loader:
            batched, batch_idx, gidx = to_gpu(batched, batch_idx, gidx)
            by = yn[gidx]
            opt.zero_grad()
            with autocast('cuda'):
                preds = m(batched, batch_idx)
                loss = crit(preds, by).mean(dim=1).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        m.eval()
        with torch.no_grad():
            va_preds, va_g = [], []
            for batched, batch_idx, gidx in va_loader:
                batched, batch_idx, gidx = to_gpu(batched, batch_idx, gidx)
                with autocast('cuda'):
                    va_preds.append(m(batched, batch_idx).float())
                va_g.append(gidx)
            vp = torch.cat(va_preds, dim=0)
            vg = torch.cat(va_g, dim=0)
            va_loss = nn.MSELoss()(vp, yn[vg])
        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP:
            print(f'  early stop @ ep {ep}', flush=True)
            break
    m.load_state_dict(best_state); m.eval()
    te_loader = make_loader(te_idx, shuffle=False)
    with torch.no_grad():
        te_preds, te_g = [], []
        for batched, batch_idx, gidx in te_loader:
            batched, batch_idx, gidx = to_gpu(batched, batch_idx, gidx)
            with autocast('cuda'):
                te_preds.append(m(batched, batch_idx).float())
            te_g.append(gidx)
        tp = torch.cat(te_preds, dim=0).cpu() * y_std + y_mean
        tg = torch.cat(te_g, dim=0).cpu()
        tt = y_all[tg]
    r2s = [r2_score(tt[:, i].numpy(), tp[:, i].numpy()) for i in range(N_OUT)]
    maes = [mean_absolute_error(tt[:, i].numpy(), tp[:, i].numpy()) for i in range(N_OUT)]
    return {'r2': r2s, 'mae': maes, 'mean_r2': float(np.mean(r2s)),
            'params': sum(p.numel() for p in m.parameters())}

CONFIGS = {'A_SMILES8': (8, 6), 'B_SMILES8_BO': (11, 6)}
results = {}
for name, (nd, ed) in CONFIGS.items():
    print(f'\n===== {name} (node {nd}d / edge {ed}d) =====', flush=True)
    results[name] = {}
    for seed in SEEDS:
        r = train_eval(nd, ed, seed)
        results[name][str(seed)] = r
        print(f'  seed {seed}: mean R² = {r["mean_r2"]:.4f}  '
              f'per-prop = ' + ' '.join(f'{p}={v:.4f}' for p, v in zip(PROPS, r['r2'])), flush=True)

with open('r4_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved -> r4_results.json')
