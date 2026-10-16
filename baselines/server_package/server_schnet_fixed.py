# -*- coding: utf-8 -*-
"""
SchNet baseline（数据修复版）

根因：原 schnet_lr_scan.py / rebuild_schnet.py / preprocess_dimenet.py 都用
      parse(..., start_idx=len(all_m))，把"累计保留的 TM 数"当成"累计块数"，
      导致 part 2/3（约 2/3 数据）的 csv_idx 系统性偏移、标签取到别的络合物。
      这解释了为什么 SchNet lr=1e-3 时 R²≈0（best_va=0.9937 ≈ 均值预测）
      且"学习率越小越好"（梯度噪声大）。

本脚本：架构与超参与原 SchNet 完全一致（20 RBF σ=0.1, atom emb 85, 5×384d CFConv,
        mean+sum pooling, MLP 768→384→192→8, bs=64, AdamW wd=1e-4, clip 1.0），
        仅做两件事：
        (1) 用正确的坐标（schnet_domain_coords.npz，全局块索引提取，元素已 100% 对齐）
            和正确的标签（new_domain_iso.pt 的 y_all）
        (2) 用与 GAT 完全相同的划分（15% held-out 13,556 + 3-fold CV × 3 seed = 9 模型）
        学习率 1e-3 —— 与 GAT 主实验同协议。
"""
import os, sys, time, json, copy, contextlib
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from torch.amp import autocast, GradScaler

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
# AMP 设置（可在训练前后统一使用）；SCHNET_FP32=1 环境变量可强制关闭 AMP
AMP_DEV_G = device.type
AMP_DTYPE_G = None if os.environ.get('SCHNET_FP32') else torch.bfloat16
print(f'AMP: {AMP_DTYPE_G if AMP_DTYPE_G else "disabled (FP32)"}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {torch.cuda.get_device_name(0)} VRAM {p.total_memory/1024**3:.1f}GB')

# ---------------- 数据 ----------------
D = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
y_all = D['y_all'].float()
N = y_all.size(0)
atoms_filt = D['atoms_filt']
coords_np = np.load('schnet_domain_coords.npz')
COORDS = coords_np['coords']; LENS = coords_np['lens']
assert len(LENS) == N and LENS.sum() == len(COORDS), f'{len(LENS)} {LENS.sum()} {len(COORDS)}'
print(f'新域 N={N:,}，原子总数 {len(COORDS):,}，平均 {len(COORDS)/N:.1f} 原子/络合物')

# ---------------- 图构建（KNN K=32，与原 SchNet 一致）----------------
K_NEIGHBORS, CUTOFF = 32, 5.0
N_RBF, RBF_SIGMA = 20, 0.1
rbf_centers = torch.linspace(0.0, CUTOFF, N_RBF)

t0 = time.time()
ZS, EIS, DS = [], [], []
off = 0
for i in range(N):
    n = int(LENS[i])
    c = torch.as_tensor(COORDS[off:off + n], dtype=torch.float32)
    off += n
    zs = torch.as_tensor(np.asarray(atoms_filt[i]), dtype=torch.long)
    dist = torch.cdist(c, c)
    k = min(K_NEIGHBORS + 1, n)
    _, nbr = torch.topk(dist, k, dim=1, largest=False)
    src, dst = [], []
    for a in range(n):
        nb = nbr[a, 1:].tolist()
        src.extend([a] * len(nb)); dst.extend(nb)
        src.extend(nb); dst.extend([a] * len(nb))
    s = torch.tensor(src, dtype=torch.long); d_ = torch.tensor(dst, dtype=torch.long)
    dd = torch.norm(c[s] - c[d_], dim=1).clamp(0, CUTOFF - 1e-6)
    ZS.append(zs); EIS.append(torch.stack([s, d_])); DS.append(dd)
    if (i + 1) % 20000 == 0:
        print(f'  ...{i+1}/{N} graphs [{time.time()-t0:.0f}s]', flush=True)
print(f'图构建完成: {N:,} 图, {sum(len(d) for d in DS)/1e6:.1f}M 边, {time.time()-t0:.0f}s')

# ---------------- 划分（与 GAT 完全一致）----------------
te_stored = np.asarray(D['te_idx'])
idx_all = np.random.RandomState(42).permutation(N)
n_test = len(te_stored)
te_idx = idx_all[:n_test].copy()
cv_idx = idx_all[n_test:].copy()
assert np.array_equal(te_idx, te_stored), 'te_idx 与 new_domain_iso.pt 存储的不一致'
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_splits = list(kf.split(cv_idx))
print(f'Test {len(te_idx):,} | CV {len(cv_idx):,}  (te_idx 与 new_domain_iso.pt 逐位一致)')

# ---------------- 模型（与原 SchNet 逐字一致）----------------
HIDDEN, N_LAYERS, N_ATOM_TYPES, DROPOUT = 384, 5, 85, 0.10

class CFConv(nn.Module):
    def __init__(self, hidden, n_rbf):
        super().__init__()
        self.filter_net = nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
    def forward(self, h, ei, rbf):
        src, dst = ei[0], ei[1]; Nn = h.size(0)
        W = self.filter_net(rbf)
        msg = h[src] * W
        out = torch.zeros(Nn, h.size(1), device=h.device, dtype=h.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return out

class SchNet(nn.Module):
    def __init__(self, hidden=HIDDEN, n_layers=N_LAYERS, n_rbf=N_RBF):
        super().__init__()
        self.embed = nn.Embedding(N_ATOM_TYPES, hidden, padding_idx=0)
        self.interactions = nn.ModuleList([CFConv(hidden, n_rbf) for _ in range(n_layers)])
        self.atomwise = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(DROPOUT)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, 8))
    def forward(self, batched):
        z, ei, rbf = batched['z'], batched['edge_index'], batched['edge_rbf']
        batch = batched['batch']; n_graphs = batched['n_graphs']
        h = self.embed(z)
        for inter, atom, norm in zip(self.interactions, self.atomwise, self.norms):
            h = norm(h + atom(inter(h, ei, rbf))); h = self.drop(h)
        mean_h = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        mean_h.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
        cnt = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
        cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
        mean_h = mean_h / cnt.clamp(min=1).unsqueeze(-1)
        sum_h = mean_h * cnt.unsqueeze(-1)
        return self.readout(torch.cat([mean_h, sum_h], dim=1))

# ---------------- 训练 ----------------
def amp_ctx():
    """AMP 上下文；SCHNET_FP32=1 时退化为 nullcontext（纯 FP32）"""
    if AMP_DTYPE_G is None:
        return contextlib.nullcontext()
    return autocast(AMP_DEV_G, dtype=AMP_DTYPE_G)

def collate(indices):
    zl, eil, dl, bl = [], [], [], []
    o = 0
    for k, i in enumerate(indices):
        n = ZS[i].size(0)
        zl.append(ZS[i]); eil.append(EIS[i] + o); dl.append(DS[i])
        bl.append(torch.full((n,), k, dtype=torch.long)); o += n
    dcat = torch.cat(dl)
    rbf = torch.exp(-(dcat[:, None] - rbf_centers.to(dcat.device)) ** 2 / (2 * RBF_SIGMA ** 2))
    # 直接同步传输（不做 stream overlap）：SchNet 是计算瓶颈，传输不是；
    # 且异步传输若不同步会让模型读到未传完的索引 -> device-side assert
    return {'z': torch.cat(zl).to(device),
            'edge_index': torch.cat(eil, dim=1).to(device),
            'edge_rbf': rbf.to(device),
            'batch': torch.cat(bl).to(device),
            'n_graphs': len(indices)}

def train_one(tr_abs, va_abs, seed, name, bs=64, epochs=200, early=40, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SchNet().to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()
    # 用 bfloat16 而非 float16：模型含 sum_h = mean_h * cnt (cnt~65) 这类大尺度中间量，
    # FP16 的动态范围(max 65504)会溢出 -> inf -> NaN；bf16 指数位与 FP32 相同，不会溢出，
    # 速度与 FP16 相当，且不需要 GradScaler。
    global AMP_DTYPE_G, AMP_DEV_G
    AMP_DTYPE = AMP_DTYPE_G = None if os.environ.get('SCHNET_FP32') else torch.bfloat16
    AMP_DEV = AMP_DEV_G = device.type
    if AMP_DTYPE is None:
        print(f'  [{name}] AMP disabled (SCHNET_FP32=1): pure FP32 + TF32', flush=True)
    scaler = None
    y_all_dev = y_all.to(device)
    y_tr = y_all_dev[tr_abs]
    ym = y_tr.mean(0); ys = y_tr.std(0).clamp(min=1e-8)
    yn = (y_all_dev - ym) / ys
    best_va, best_state, pat = float('inf'), None, 0
    n_tr = len(tr_abs)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n_tr)
        tot, nb = 0.0, 0
        for s in range(0, n_tr, bs):
            bi = tr_abs[perm[s:s + bs]].tolist()
            batched = collate(bi)
            opt.zero_grad()
            with amp_ctx():
                loss = crit(model(batched), yn[bi])
            if not torch.isfinite(loss):
                # 自适应退避：若是优化不稳（而非精度）导致的非有限 loss，减半学习率后继续
                for gp in opt.param_groups:
                    gp['lr'] *= 0.5
                print(f'  [{name}] WARN: non-finite loss at ep{ep+1}; batch skipped, '
                      f'lr -> {opt.param_groups[0]["lr"]:.2e}', flush=True)
                opt.zero_grad(); continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                for gp in opt.param_groups:
                    gp['lr'] *= 0.5
                print(f'  [{name}] WARN: non-finite grad-norm at ep{ep+1}; step skipped, '
                      f'lr -> {opt.param_groups[0]["lr"]:.2e}', flush=True)
                opt.zero_grad(); continue
            opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vp = []
            for s in range(0, len(va_abs), bs):
                bi = va_abs[s:s + bs].tolist()
                with amp_ctx(): vp.append(model(collate(bi)).float())
            vp = torch.cat(vp)
            va = crit(vp, yn[va_abs]).item()
        if va < best_va:
            best_va = va; best_state = {k: v.clone() for k, v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
        if (ep + 1) % 10 == 0 or ep < 5:
            # 顺带报 val 的原始尺度 mean R²，便于早期判断数据修复是否生效
            vpo = vp.cpu().numpy() * ys.cpu().numpy() + ym.cpu().numpy()
            vt = y_all[va_abs].numpy()
            if np.isfinite(vpo).all():
                vr2 = float(np.mean([r2_score(vt[:, j], vpo[:, j]) for j in range(8)]))
            else:
                vr2 = float('nan')
                print(f'  [{name}] WARN: non-finite val predictions at ep{ep+1}', flush=True)
            print(f'  [{name}] ep{ep+1:3d} tr={tot/max(1,nb):.4f} va={va:.4f} valR2={vr2:+.4f} pat={pat}', flush=True)
        if pat >= early:
            print(f'  [{name}] early stop @ ep{ep+1}', flush=True); break
    if best_state is None:
        print(f'  [{name}] WARN: no finite validation epoch; using final weights', flush=True)
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best_state, ym.cpu(), ys.cpu()

SEEDS = [42, 123, 456]
CKPT = 'ensemble_checkpoint_SchNet.pt'
all_models, log = [], []
done = set()
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    all_models, log = ck['models'], ck.get('log', [])
    done = {(m['fold'], m['seed']) for m in all_models}
    print(f'Restored {len(all_models)} models')

t_start = time.time()
for fi in range(3):
    tr = cv_idx[fold_splits[fi][0]]; va = cv_idx[fold_splits[fi][1]]
    for seed in SEEDS:
        if (fi, seed) in done:
            print(f'  SKIP Fold{fi}_Seed{seed}'); continue
        name = f'Fold{fi}_Seed{seed}'
        t1 = time.time()
        print(f'  Training {name} (train={len(tr):,} val={len(va):,})...', flush=True)
        st, ym, ys = train_one(tr, va, seed, name)
        el = (time.time() - t1) / 60
        all_models.append({'state': st, 'y_mean': ym, 'y_std': ys, 'fold': fi, 'seed': seed})
        log.append({'fold': fi, 'seed': seed, 'min': round(el, 1)})
        done.add((fi, seed))
        print(f'  Done {name} in {el:.1f} min', flush=True)
        torch.save({'models': all_models, 'log': log}, CKPT)

# ---------------- 测试集评估 ----------------
print('Test eval...', flush=True)
preds = []
for m in all_models:
    model = SchNet().to(device)
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in m['state'].items()})
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(te_idx), 64):
            bi = te_idx[s:s + 64].tolist()
            with amp_ctx(): out.append(model(collate(bi)).float().cpu())
    out = torch.cat(out).numpy() * m['y_std'].numpy() + m['y_mean'].numpy()
    preds.append(out)
    del model; torch.cuda.empty_cache()
P = np.stack(preds, 0)
yt = y_all[te_idx].numpy()
ens = P.mean(0)
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
res = {'variant': 'SchNet_fixed'}
for i, p in enumerate(targets):
    res[p] = {'ens_r2': float(r2_score(yt[:, i], ens[:, i])),
              'mean_r2': float(np.mean([r2_score(yt[:, i], P[m, :, i]) for m in range(len(P))])),
              'std_r2': float(np.std([r2_score(yt[:, i], P[m, :, i]) for m in range(len(P))])),
              'ens_mae': float(mean_absolute_error(yt[:, i], ens[:, i]))}
res['mean_ens_r2'] = float(np.mean([res[p]['ens_r2'] for p in targets]))
np.savez('test_preds_SchNet.npz', preds=P, ens_mean=ens, y_true=yt, te_idx=te_idx)
json.dump(res, open('results_SchNet.json', 'w'), indent=2)
print(f"\nSchNet mean ensemble R2 = {res['mean_ens_r2']:.4f}")
print(json.dumps(res, indent=2)[:900])
print(f'\nALL DONE in {(time.time()-t_start)/60:.0f} min')
