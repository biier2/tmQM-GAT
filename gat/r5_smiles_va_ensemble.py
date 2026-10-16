# -*- coding: utf-8 -*-
"""R5: SMILES 集成在论文主域 (vA, 90,317) 重训 — 意见6 方案B
与 R3 (r3_smiles_ensemble.py) 协议逐字一致, 唯一区别:
  - 域: vA_domain_smiles.pt (本地论文管线逐字同逻辑构建的 vA 域 90,317 图, 上传后直接加载)。
        2026-09-02 实测: 服务器 r2b_graphs.pt 前 90,317 ≠ 论文主域行序 (rdkit 版本 ±19 错位),
        故弃用截断路线, 改为本地构建数据包 (已自检 y_true 与 3D 集成逐位一致)。
  - 划分: 直接用 paper_domain_vA/ensemble_split.npz 的 te_idx / cv_idx / fold_0-2_train/val
          (与 3D 集成 ensemble_cv_weighted 完全相同的划分 → 同池同划分配对比较最严格)
  - 动机: R3 在 95,152 域自划测试集 (RandomState(42), 14,272), 实测 62% 的 vA 测试集
          (13,547) 落在 R3 训练池 → R3 模型在 13,547 上推断有训练泄漏, 不能作主结果。
协议: 3-fold × 3-seed = 9 模型, AdamW lr=1e-3 wd=1e-4, CosineAnnealingWarmRestarts(T_0=50,T_mult=2),
      AMP, bs=256, epochs=200, early_stop=40, sqrt 金属权重, per-model local 标准化
输出: r5_smiles_va_ckpt.pt (断点) + r5_smiles_va_ensemble.npz + r5_smiles_va_results.json
环境变量: R5_NW (DataLoader 进程数, 默认 8); R5_SMOKE=1 冒烟测试 (2 epochs, 不保存)
"""
import json, os, sys, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
print(f'Device: {device}', flush=True)

PROPS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10
SEEDS = [42, 123, 456]; FOLDS = 3
BS = 256                                       # 协议固定 bs=256
NW = int(os.environ.get('R5_NW', '8'))         # DataLoader 进程数 (16 核给 8)
SMOKE = os.environ.get('R5_SMOKE', '0') == '1'
N_VALID = 90317                                # vA 域大小 (README: 前 90,317 保留分子)
CKPT = 'r5_smiles_va_ckpt.pt'

# ---------- 数据: vA_domain_smiles.pt (本地论文管线构建的 vA 域 90,317 图) ----------
data = torch.load('vA_domain_smiles.pt', weights_only=False)
graphs = data['graphs']; y_all = data['y_all']; atoms_filt = data['atoms_filt']
print(f'加载 vA_domain_smiles.pt: {len(graphs):,} 图', flush=True)
assert len(graphs) == N_VALID, f'图数 {len(graphs):,} != 90,317, vA_domain_smiles.pt 版本不对'

# ---------- 划分: 直接用论文主域 ensemble_split.npz (与 3D 集成相同) ----------
split_paths = ['ensemble_split.npz', 'data_tmQM/paper_domain_vA/ensemble_split.npz']
sp = None
for p in split_paths:
    if os.path.exists(p):
        sp = np.load(p); print(f'划分文件: {p}', flush=True); break
assert sp is not None, '未找到 ensemble_split.npz (试过: ' + ', '.join(split_paths) + ')'
te_idx = sp['te_idx'].copy()
cv_idx = sp['cv_idx'].copy()
fold_splits = [ (sp[f'fold_{f}_train'], sp[f'fold_{f}_val']) for f in range(3) ]
assert len(te_idx) == 13547, f'te_idx 长度 {len(te_idx)} != 13,547'
assert te_idx.max() < N_VALID and cv_idx.max() < N_VALID, '划分索引超出 90,317 域'
print(f'固定测试 {len(te_idx):,} / CV 池 {len(cv_idx):,} (论文主域 vA)', flush=True)

# ---------- 对齐自检: 与 3D 集成 (ensemble_predictions.npz) 逐位核对 te_idx + y_true ----------
# 目的: 防止服务器 rdkit 版本差异导致保留序列错位 (README: 差 19 个分子可能错位几个)
ep_path = 'ensemble_predictions.npz'
if os.path.exists(ep_path):
    ep = np.load(ep_path)
    same_te = np.array_equal(ep['te_idx'], te_idx)
    yt_ref = ep['y_true']
    yt_here = y_all[te_idx].numpy()
    same_yt = np.allclose(yt_here, yt_ref, atol=1e-4)
    print(f'对齐自检 vs ensemble_predictions.npz: te_idx 相同={same_te}, y_true 相同={same_yt}', flush=True)
    if not (same_te and same_yt):
        print('!! 对齐失败: vA_domain_smiles.pt 与 3D 集成的行序不一致 (数据包传错/损坏?)。中止。', flush=True)
        sys.exit(1)
else:
    print('警告: 未找到 ensemble_predictions.npz, 跳过对齐自检', flush=True)

# ---------- 金属权重 sqrt(N/n) — 在 vA 域上计算, 与 3D 集成一致 ----------
mc = Counter()
for atoms in atoms_filt:
    for z in set(atoms):
        if z >= 21: mc[z] += 1
total_c = len(atoms_filt)
weight_map = {z: max(0.5, min(3.0, (total_c / c) ** 0.5)) for z, c in mc.items()}
sample_weights = torch.ones(len(graphs))
for i, atoms in enumerate(atoms_filt):
    w = max((weight_map.get(z, 1.0) for z in set(atoms) if z >= 21), default=1.0)
    sample_weights[i] = w
sample_weights = sample_weights / sample_weights.mean()

# ---------- 模型 (与 R3 完全相同: XYZ_GNN, node 8d / edge 6d 截断) ----------
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

class XYZ_GNN(nn.Module):
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
            nn.Linear(HIDDEN // 2, 8))
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

# ---------- 数据管道 (与 R3 相同: numpy 向量化 collate + 多进程) ----------
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
    kw = dict(batch_size=BS, shuffle=shuffle, collate_fn=collate_fn,
              num_workers=NW, pin_memory=True, drop_last=False)
    if NW > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(Subset(GraphDataset(graphs), indices), **kw)

def to_gpu(batched, batch_idx, gidx):
    return ({k: v.to(device, non_blocking=True) for k, v in batched.items()},
            batch_idx.to(device, non_blocking=True),
            gidx.to(device, non_blocking=True))

# ---------- train_one_model (协议同 R3/3D 集成) ----------
def train_one_model(train_idx, val_idx, seed, fold_name):
    torch.manual_seed(seed); np.random.seed(seed)
    model = XYZ_GNN().to(device)
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    crit = nn.MSELoss(reduction='none')
    scaler = GradScaler('cuda')
    y_mean_local = y_all[train_idx].mean(0)
    y_std_local = y_all[train_idx].std(0).clamp(min=1e-8)
    y_norm = (y_all - y_mean_local) / y_std_local
    yn = y_norm.to(device)
    sw = sample_weights.to(device)
    best_va, best_state, patience = float('inf'), None, 0
    epochs = 2 if SMOKE else 200
    early_stop = 40
    tr_loader = make_loader(train_idx, shuffle=True)
    va_loader = make_loader(val_idx, shuffle=False)
    print(f'  [{fold_name}] epochs={epochs} train={len(train_idx):,} val={len(val_idx):,} bs={BS}', flush=True)
    for ep in range(epochs):
        model.train()
        for batched, batch_idx, gidx in tr_loader:
            batched, batch_idx, gidx = to_gpu(batched, batch_idx, gidx)
            by = yn[gidx]
            opt.zero_grad()
            with autocast('cuda'):
                preds = model(batched, batch_idx)
                loss = (crit(preds, by).mean(dim=1) * sw[gidx]).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sched.step()
        model.eval()
        with torch.no_grad():
            va_preds, va_g = [], []
            for batched, batch_idx, gidx in va_loader:
                batched, batch_idx, gidx = to_gpu(batched, batch_idx, gidx)
                with autocast('cuda'):
                    va_preds.append(model(batched, batch_idx).float())
                va_g.append(gidx)
            vp = torch.cat(va_preds, dim=0)
            vg = torch.cat(va_g, dim=0)
            va_loss = nn.MSELoss()(vp, yn[vg])
        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= early_stop and not SMOKE:
            print(f'  early stop @ ep {ep}', flush=True)
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        va_preds, va_g = [], []
        for batched, batch_idx, gidx in va_loader:
            batched, batch_idx, gidx = to_gpu(batched, batch_idx, gidx)
            with autocast('cuda'):
                va_preds.append(model(batched, batch_idx).float())
            va_g.append(gidx)
        vp = torch.cat(va_preds, dim=0)
        vg = torch.cat(va_g, dim=0).cpu()
        vp_orig = vp.cpu() * y_std_local + y_mean_local
        yv_orig = y_all[vg]
        val_r2s = [r2_score(yv_orig[:, i].numpy(), vp_orig[:, i].numpy()) for i in range(8)]
    return model.state_dict(), y_mean_local, y_std_local, val_r2s

# ---------- 主循环 (断点续训) ----------
all_models = []
results_log = []
completed = set()
if os.path.exists(CKPT):
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    all_models = ckpt['models']; results_log = ckpt.get('log', [])
    for m in all_models:
        completed.add((m['fold'], m['seed']))
    print(f'恢复 {len(all_models)} 模型, 已完成 {sorted(completed)}', flush=True)

t_start = time.time()
for fold_idx in range(FOLDS):
    tr_abs = fold_splits[fold_idx][0]
    va_abs = fold_splits[fold_idx][1]
    for seed in SEEDS:
        if (fold_idx, seed) in completed:
            print(f'SKIP Fold{fold_idx}_Seed{seed}', flush=True)
            continue
        name = f'Fold{fold_idx}_Seed{seed}'
        print(f'\n===== {name} =====', flush=True)
        t0 = time.time()
        state, ym, ys, val_r2s = train_one_model(tr_abs, va_abs, seed, name)
        elapsed = time.time() - t0
        all_models.append({'state': state, 'y_mean': ym, 'y_std': ys,
                           'fold': fold_idx, 'seed': seed, 'val_r2s': val_r2s})
        results_log.append({'fold': fold_idx, 'seed': seed,
                            'val_mean_r2': float(np.mean(val_r2s)), 'val_r2s': val_r2s,
                            'time_min': elapsed / 60})
        completed.add((fold_idx, seed))
        print(f'  完成 {elapsed/60:.1f}min, Val Mean R² = {np.mean(val_r2s):.4f}', flush=True)
    if not SMOKE:
        torch.save({'models': all_models, 'log': results_log}, CKPT)
        print(f'Checkpoint saved ({len(all_models)}/9)', flush=True)

if SMOKE:
    print('冒烟测试完成, 不保存 (R5_SMOKE=1)', flush=True)
    sys.exit(0)

print(f'\n训练完成 {len(all_models)}/9 模型', flush=True)

# ---------- 测试集评估 (9 成员 + 集成) ----------
print('\n测试集评估 (13,547)...', flush=True)
te_loader = make_loader(te_idx, shuffle=False)
per_model = np.zeros((len(all_models), len(te_idx), 8), dtype=np.float32)
for mi, m in enumerate(all_models):
    model = XYZ_GNN().to(device)
    model.load_state_dict(m['state']); model.eval()
    y_mean_local = m['y_mean']; y_std_local = m['y_std']
    preds = []
    with torch.no_grad():
        for batched, batch_idx, gidx in te_loader:
            batched, batch_idx, gidx = to_gpu(batched, batch_idx, gidx)
            with autocast('cuda'):
                preds.append(model(batched, batch_idx).float().cpu())
    p = torch.cat(preds, dim=0) * y_std_local + y_mean_local
    per_model[mi] = p.numpy()
    print(f'  成员 {mi+1}/9 完成 (fold {m["fold"]}, seed {m["seed"]})', flush=True)

y_true = y_all[te_idx].numpy()
ens_mean = per_model.mean(axis=0)
ens_std = per_model.std(axis=0)

r2_single = {PROPS[i]: [float(r2_score(y_true[:, i], per_model[m, :, i])) for m in range(len(all_models))] for i in range(8)}
r2_ens = {PROPS[i]: float(r2_score(y_true[:, i], ens_mean[:, i])) for i in range(8)}
mae_ens = {PROPS[i]: float(mean_absolute_error(y_true[:, i], ens_mean[:, i])) for i in range(8)}
mean_r2_ens = float(np.mean(list(r2_ens.values())))

print('\nSMILES 集成 vA 域 (9 成员均值, 13,547 测试) 逐性质:')
for i, p in enumerate(PROPS):
    print(f'  {p:14s} R² = {r2_ens[p]:.4f}  MAE = {mae_ens[p]:.4f}  '
          f'(单模型 {np.mean(r2_single[p]):.4f} ± {np.std(r2_single[p]):.4f})')
print(f'  Mean R² = {mean_r2_ens:.4f}')

np.savez_compressed('r5_smiles_va_ensemble.npz',
                    per_model=per_model, mean=ens_mean, std=ens_std,
                    y_true=y_true, te_idx=te_idx)
with open('r5_smiles_va_results.json', 'w') as f:
    json.dump({'r2_ens': r2_ens, 'mae_ens': mae_ens, 'mean_r2': mean_r2_ens,
               'r2_single': r2_single, 'log': results_log}, f, indent=2)
print('Saved -> r5_smiles_va_ensemble.npz + r5_smiles_va_results.json')
