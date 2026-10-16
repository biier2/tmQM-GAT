# -*- coding: utf-8 -*-
"""
DimeNet++ baseline（数据修复版）

根因与 SchNet 完全相同：原 preprocess_dimenet.py 用
    parse_one(..., start_idx=len(all_m))      # ✗ all_m 是"累计保留的 TM 数"，不是"累计块数"
导致 part 2/3（约 2/3 数据）的 csv_idx 系统性偏移、标签取到别的络合物。
这正是 old train log 的典型标签噪声特征：train_rmse 一路降到 0.35（模型在记忆 2/3 的随机
标签），而 val_rmse 260 个 epoch 死死钉在 0.956（≈ 均值预测）。

本脚本：架构与超参与原 DimeNet++ 完全一致（PyG DimeNetPlusPlus: hidden 128, 4 interaction
        blocks, int_emb 64, basis_emb 8, out_emb 256, num_spherical 7, num_radial 6,
        cutoff 5.0 Å, envelope 5, 3 层输出 MLP；半径图 r=5.0 / 每节点 ≤32 近邻；
        bs=8, AdamW wd=1e-4, clip 5.0），仅做两件事：
        (1) 用正确的坐标（schnet_domain_coords.npz，全局块索引提取，元素 100% 对齐）
            和正确的标签（new_domain_iso.pt 的 y_all）
        (2) 与 SchNet 修复版 / GAT 四变体相同的域与划分

⚠️ 运行时长：实测约 220 s/epoch（51,214 训练 + 验证）。默认 --epochs 300（= 撤稿 pilot 的
   原协议，warmup 10 + cosine 1e-3→1e-5 + 不早停），单模型 ≈ 18.3 h。
   默认 --folds 0 --seeds 42 123 456 = 3 个模型 ≈ 55 小时。
   要 9 模型（与 SchNet/GAT 变体一致的 3 折 × 3 种子）用 --folds 0 1 2，约 165 小时。
   ★ 可分两段跑：先 --folds 0（3 模型），之后用 --folds 0 1 2 重跑会自动跳过已完成的
     (fold, seed) 只补余下 6 个，最后统一评估全部模型 —— 不会重复训练。

用法：
    python server_dimenet_fixed.py --smoke                 # 2 分钟冒烟
    python server_dimenet_fixed.py --subset 2000 --epochs 60   # ~40 分钟小样本拟合诊断
    nohup python server_dimenet_fixed.py > train_dimenet_fixed.log 2>&1 &
"""
import os, sys, time, json, argparse, contextlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader
from torch_geometric.nn import DimeNetPlusPlus

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']

ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=300,
                help='默认 300 = 撤稿 pilot 的原协议；cosine 调度按此值展开，改小会截断退火')
ap.add_argument('--batch-size', type=int, default=8, help='原 DimeNet++ 用 8；16GB 卡可试 16')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--weight-decay', type=float, default=1e-4)
ap.add_argument('--warmup', type=int, default=10)
ap.add_argument('--min-lr', type=float, default=1e-5)
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--blocks', type=int, default=4)
ap.add_argument('--folds', type=int, nargs='+', default=[0], help='用哪些 fold（默认只有 0）')
ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
ap.add_argument('--val-every', type=int, default=5, help='每 N epoch 评估一次验证集（省时间）')
ap.add_argument('--workers', type=int, default=0, help='0=主进程加载，避免 CUDA fork 死锁')
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--subset', type=int, default=0, help='诊断：只用前 N 个训练分子')
args = ap.parse_args()
# 全程 FP32 + TF32：PyG 稀疏算子与 autocast 组合风险高（SchNet 那次 fp16 溢出就是教训），
# 且原 DimeNet++ 实测就是 FP32（220 s/epoch），换成 AMP 收益不确定。

if args.smoke:
    args.epochs, args.batch_size, args.val_every = 2, 4, 1
    args.folds, args.seeds = [0], [42]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM {p.total_memory/1024**3:.1f}GB')
print(f'配置: epochs={args.epochs} bs={args.batch_size} lr={args.lr} folds={args.folds} seeds={args.seeds}')

# ---------------- 数据（与 server_schnet_fixed.py 逐行同源）----------------
D = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
y_all = D['y_all'].float()
N = y_all.size(0)
atoms_filt = D['atoms_filt']
coords_np = np.load('schnet_domain_coords.npz')
COORDS = coords_np['coords']; LENS = coords_np['lens']
assert len(LENS) == N and LENS.sum() == len(COORDS), f'{len(LENS)} {LENS.sum()} {len(COORDS)}'
print(f'域 N={N:,}，原子总数 {len(COORDS):,}，平均 {len(COORDS)/N:.1f} 原子/络合物')

# 一次性切出每个络合物的 pos / z（避免每 epoch 重切）
POS, ZS = [], []
off = 0
for i in range(N):
    n = int(LENS[i])
    POS.append(torch.as_tensor(COORDS[off:off + n], dtype=torch.float32))
    off += n
    ZS.append(torch.as_tensor(np.asarray(atoms_filt[i]), dtype=torch.long))
print(f'pos/z 切分完成（{time.time():.0f}）')

# ---------------- 划分（与 GAT 变体 / SchNet 相同）----------------
te_stored = np.asarray(D['te_idx'])
idx_all = np.random.RandomState(42).permutation(N)
n_test = len(te_stored)
te_idx = idx_all[:n_test].copy()
cv_idx = idx_all[n_test:].copy()
assert np.array_equal(te_idx, te_stored), 'te_idx 与 new_domain_iso.pt 存储的不一致'
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_splits = list(kf.split(cv_idx))
print(f'Test {len(te_idx):,} | CV {len(cv_idx):,}')

# ---------------- 模型 ----------------
def build_model():
    return DimeNetPlusPlus(
        hidden_channels=args.hidden, out_channels=len(TARGETS), num_blocks=args.blocks,
        int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, cutoff=5.0, max_num_neighbors=32,
        envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3)

_m = build_model()
print(f'DimeNetPlusPlus: hidden={args.hidden} blocks={args.blocks} out=8  '
      f'params={sum(p.numel() for p in _m.parameters())/1e6:.2f}M')
del _m

def make_loader(sel, tag, shuffle, bs):
    """按索引构 Data 列表。z-score 用训练集统计（评估时反标准化）。"""
    data = []
    for k in sel:
        data.append(Data(z=ZS[k], pos=POS[k],
                         y=((y_all[k] - ym_cpu) / ys_cpu).view(1, -1)))
    print(f'  [{tag}] {len(data):,} 分子', flush=True)
    return DataLoader(data, batch_size=bs, shuffle=shuffle, num_workers=args.workers)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, ys = [], []
    for d in loader:
        d = d.to(device)
        preds.append(model(d.z, d.pos, d.batch)); ys.append(d.y)
    return torch.cat(preds), torch.cat(ys)

def train_one(tr_abs, va_abs, seed, name):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - args.warmup), eta_min=args.min_lr)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup])

    tr_loader = make_loader(tr_abs, f'{name}/train', True, args.batch_size)
    va_loader = make_loader(va_abs, f'{name}/val', False, args.batch_size)

    best_va, best_state = float('inf'), None
    for ep in range(args.epochs):
        t_ep = time.time()
        model.train(); tot, n = 0.0, 0
        for d in tr_loader:
            d = d.to(device)
            opt.zero_grad()
            # DimeNet++ 前向内部自建半径图（与 PyG 参考实现一致）
            loss = F.mse_loss(model(d.z, d.pos, d.batch), d.y)
            if not torch.isfinite(loss):
                for gp in opt.param_groups: gp['lr'] *= 0.5
                print(f'  [{name}] WARN: non-finite loss ep{ep+1}; batch skipped, '
                      f'lr -> {opt.param_groups[0]["lr"]:.2e}', flush=True)
                opt.zero_grad(); continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gn):
                for gp in opt.param_groups: gp['lr'] *= 0.5
                print(f'  [{name}] WARN: non-finite grad ep{ep+1}; step skipped', flush=True)
                opt.zero_grad(); continue
            opt.step()
            tot += loss.item() * d.num_graphs; n += d.num_graphs
        sched.step()
        tr_rmse = (tot / max(1, n)) ** 0.5
        if (ep + 1) % args.val_every == 0 or ep + 1 == args.epochs:
            vp, vy = evaluate(model, va_loader)
            va = float(F.mse_loss(vp, vy).sqrt())
            if np.isfinite(va) and va < best_va:
                best_va = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep < 3:
                p = vp.cpu().numpy() * ys_cpu.numpy() + ym_cpu.numpy()
                t = vy.cpu().numpy() * ys_cpu.numpy() + ym_cpu.numpy()
                vr2 = float(np.mean([r2_score(t[:, j], p[:, j]) for j in range(8)])) \
                    if np.isfinite(p).all() else float('nan')
                print(f'  [{name}] ep{ep+1:3d} tr={tr_rmse:.4f} va={va:.4f} valR2={vr2:+.4f} '
                      f'{time.time()-t_ep:.0f}s', flush=True)
        elif (ep + 1) % 10 == 0:
            print(f'  [{name}] ep{ep+1:3d} tr={tr_rmse:.4f} {time.time()-t_ep:.0f}s', flush=True)
    if best_state is None:
        print(f'  [{name}] WARN: no finite validation epoch; using final weights', flush=True)
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    del model, tr_loader, va_loader
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best_state

# ---------------- 主循环 ----------------
ym_cpu = y_all[cv_idx].mean(0)          # 占位，每个模型内用训练集统计重算
ys_cpu = y_all[cv_idx].std(0).clamp(min=1e-8)

# ⚠️ 冒烟必须用独立 checkpoint：--smoke 训的是"128 分子 × 2 epoch"的废模型，若写进正式
# checkpoint，正式跑一启动就会 "Restored 1 models / SKIP Fold0_Seed42"，把这个废模型当成
# 一个正式集成成员（2026-09-18 实际踩过，3 模型集成被拖到 0.832）。
CKPT = 'smoke_checkpoint_DimeNet.pt' if args.smoke else 'ensemble_checkpoint_DimeNet.pt'
all_models, log = [], []
done = set()
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    all_models, log = ck['models'], ck.get('log', [])
    done = {(m['fold'], m['seed']) for m in all_models}
    print(f'Restored {len(all_models)} models')

t_start = time.time()
for fi in args.folds:
    tr_full = cv_idx[fold_splits[fi][0]]; va_full = cv_idx[fold_splits[fi][1]]
    if args.smoke:
        tr_full, va_full = tr_full[:128], va_full[:32]
    elif args.subset:
        tr_full = tr_full[:args.subset]; va_full = va_full[:max(64, args.subset // 8)]
    for seed in args.seeds:
        if (fi, seed) in done:
            print(f'  SKIP Fold{fi}_Seed{seed}'); continue
        name = f'Fold{fi}_Seed{seed}'
        # z-score 统计量：每个模型用自己训练集的（与 SchNet/GAT 一致）
        ym_cpu = y_all[tr_full].mean(0)
        ys_cpu = y_all[tr_full].std(0).clamp(min=1e-8)
        t1 = time.time()
        print(f'  Training {name} (train={len(tr_full):,} val={len(va_full):,})...', flush=True)
        st = train_one(tr_full, va_full, seed, name)
        el = (time.time() - t1) / 60
        all_models.append({'state': st, 'y_mean': ym_cpu.clone(), 'y_std': ys_cpu.clone(),
                           'fold': fi, 'seed': seed})
        log.append({'fold': fi, 'seed': seed, 'min': round(el, 1)})
        done.add((fi, seed))
        print(f'  Done {name} in {el:.1f} min', flush=True)
        torch.save({'models': all_models, 'log': log}, CKPT)

# ---------------- 测试集评估 ----------------
print('Test eval...', flush=True)
preds = []
for m in all_models:
    model = build_model().to(device)
    model.load_state_dict(m['state']); model.eval()
    data = [Data(z=ZS[i], pos=POS[i]) for i in te_idx]
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    out = []
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            out.append(model(d.z, d.pos, d.batch).float().cpu())
    out = torch.cat(out).numpy() * m['y_std'].numpy() + m['y_mean'].numpy()
    preds.append(out)
    del model, loader, data
    if torch.cuda.is_available(): torch.cuda.empty_cache()
P = np.stack(preds, 0)
yt = y_all[te_idx].numpy()
ens = P.mean(0)
res = {'variant': 'DimeNet_fixed', 'n_models': len(P), 'folds': args.folds, 'seeds': args.seeds}
for i, p in enumerate(TARGETS):
    res[p] = {'ens_r2': float(r2_score(yt[:, i], ens[:, i])),
              'mean_r2': float(np.mean([r2_score(yt[:, i], P[m, :, i]) for m in range(len(P))])),
              'std_r2': float(np.std([r2_score(yt[:, i], P[m, :, i]) for m in range(len(P))])),
              'ens_mae': float(mean_absolute_error(yt[:, i], ens[:, i]))}
res['mean_ens_r2'] = float(np.mean([res[p]['ens_r2'] for p in TARGETS]))
np.savez('test_preds_DimeNet.npz', preds=P, ens_mean=ens, y_true=yt, te_idx=te_idx)
json.dump(res, open('results_DimeNet.json', 'w'), indent=2)
print(f"\nDimeNet++ mean ensemble R2 = {res['mean_ens_r2']:.4f}")
print(json.dumps(res, indent=2)[:900])
print(f'\nALL DONE in {(time.time()-t_start)/60:.0f} min')
