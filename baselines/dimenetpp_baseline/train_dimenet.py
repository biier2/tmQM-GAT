# -*- coding: utf-8 -*-
"""DimeNet++ 受控基线 — 训练/评估 (服务器 GPU)

协议 (与论文严格对齐, 详见 README.md):
  - 测试集: ensemble_split.npz te_idx (13,547, 不变) — 与 GNN 同一测试划分
  - 训练/验证 = 论文折0划分 (fold_0_train 51,180 / fold_0_val 25,590, 与 3D GAT 折0 同集)
  - 8 性质多任务 (DimeNetPlusPlus out_channels=8), z-score 训练集统计
  - 3D 输入: tmQM 原始 DFT XYZ; 半径图 r=5.0 Å / 每节点 32 近邻

用法:
  python train_dimenet.py --smoke          # 2 epoch 冒烟 (约 2 分钟, 验证管线)
  python train_dimenet.py --subset 2000 --epochs 150 --patience 200   # 小样本拟合诊断 (约 40 分钟)
  python train_dimenet.py --epochs 300 --val-every 5   # R4 全量训练 (cosine 调度, batch 8)
  python train_dimenet.py --resume         # 断点续训 (从 dimenetpp_last.pt)
  python train_dimenet.py --schedule plateau --epochs 200 --batch-size 16 --lr 5e-4

输出: dimenetpp_best.pt / dimenetpp_last.pt / dimenet_train_log.csv / dimenet_test_preds.npz
"""
import os, sys, csv, time, argparse
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader
from torch_geometric.nn import DimeNetPlusPlus

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=os.path.join(SCRIPT_DIR, 'dimenet_data.npz'))
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--blocks', type=int, default=4)
parser.add_argument('--workers', type=int, default=0, help='0=主进程加载(推荐, 避免 CUDA fork 死锁; 数据加载不是瓶颈)')
parser.add_argument('--patience', type=int, default=40, help='plateau 调度的早停 patience (cosine 下不早停)')
parser.add_argument('--schedule', choices=['cosine', 'plateau'], default='cosine',
                    help='cosine: 全程余弦退火 (R4, 不受 val 平台期影响); plateau: 旧版按 val 衰减')
parser.add_argument('--warmup', type=int, default=10, help='cosine 预热 epoch 数')
parser.add_argument('--min-lr', type=float, default=1e-5, help='cosine 末端学习率')
parser.add_argument('--val-every', type=int, default=1, help='每 N 个 epoch 评估一次验证集 (N>1 省约 100s/epoch)')
parser.add_argument('--smoke', action='store_true', help='2 epoch 冒烟测试')
parser.add_argument('--subset', type=int, default=0, help='诊断: 只用前 N 个训练分子 (0=全量), 用于快速判断模型能否拟合小样本')
parser.add_argument('--resume', action='store_true', help='从 dimenetpp_last.pt 续训')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ===== 数据加载 =====
print('Loading dimenet_data.npz ...')
D = np.load(args.data, allow_pickle=True)
y_mean_cpu = torch.from_numpy(D['y_mean'].astype(np.float32))
y_std_cpu = torch.from_numpy(D['y_std'].astype(np.float32))
y_mean = y_mean_cpu.to(device)
y_std = y_std_cpu.to(device)
print(f'y_mean={y_mean.cpu().numpy().round(2)}', flush=True)
print(f'y_std ={y_std.cpu().numpy().round(2)}', flush=True)

def make_dataset(sel, tag):
    """按索引序列加载 z/pos/edge_index/y, 组装 PyG Data 列表。

    ⚠️ NpzFile (np.load 返回对象) 不缓存成员: D['z_train'] 每访问一次就重新
    解压+反序列化整列对象数组。必须在循环外取一次, 否则 O(N) 次解压整列,
    组装会慢上几个小时。"""
    z_arr = D[f'z_{tag}']
    pos_arr = D[f'pos_{tag}']
    edge_arr = D[f'edge_{tag}']
    y_arr = D[f'y_{tag}']
    data = []
    for k in sel:
        data.append(Data(
            z=torch.from_numpy(z_arr[k].astype(np.int64)),
            pos=torch.from_numpy(pos_arr[k]),
            edge_index=torch.from_numpy(edge_arr[k].astype(np.int64)),
            # z-score(训练集统计) — 与论文 GNN 管线一致; 评估时再乘 y_std 加 y_mean 反标准化。
            # (1, 8): PyG 默认把 y 当节点级属性沿 dim0 拼接, (8,) 会拼成 (batch*8,) 与图级输出 (batch, 8) 对不上
            y=((torch.from_numpy(y_arr[k]) - y_mean_cpu) / y_std_cpu).view(1, -1)))
    return data

sel_tr, sel_va, sel_te = range(D['train_sel'].shape[0]), range(D['val_sel'].shape[0]), range(D['test_sel'].shape[0])
if args.smoke:
    print('[smoke] 数据集缩减: train 128 / val 32 / test 64, epochs=2, batch=4')
    args.epochs, args.batch_size, args.patience = 2, 4, 100
    sel_tr, sel_va, sel_te = range(128), range(32), range(64)
    args.lr = 5e-4       # 冒烟用小学习率快收敛
    args.schedule = 'plateau'   # 2 epoch 冒烟不适用 cosine 预热
if args.subset and not args.smoke:
    n_sub = args.subset
    print(f'[subset] 诊断模式: train {n_sub} / val {max(64, n_sub // 8)} / test 64, epochs={args.epochs}, '
          f'判定: train_rmse < 0.6 = 架构健康(问题在训练规模/调度); 卡在 0.9+ = 几何通路失效')
    sel_tr, sel_va, sel_te = range(n_sub), range(max(64, n_sub // 8)), range(64)
t_ds = time.time()
train_ds = make_dataset(sel_tr, 'train')
val_ds = make_dataset(sel_va, 'val')
test_ds = make_dataset(sel_te, 'test')
print(f'Train {len(train_ds)} / Val {len(val_ds)} / Test {len(test_ds)}  (数据集组装 {time.time()-t_ds:.0f}s)', flush=True)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# ===== 模型 =====
model = DimeNetPlusPlus(
    hidden_channels=args.hidden, out_channels=len(TARGETS), num_blocks=args.blocks,
    int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
    num_spherical=7, num_radial=6, cutoff=float(D['cutoff']),
    max_num_neighbors=int(D['max_neighbors']),
    envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'DimeNetPlusPlus: hidden={args.hidden} blocks={args.blocks} out=8  params={n_params/1e6:.2f}M', flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.schedule == 'cosine':
    # R4 关键修复: R3 用 ReduceLROnPlateau, val 长期停在 0.95 均值预测地板 → lr 被一路
    # 衰减到 1e-5, 训练还没开始拟合就被冻结 (epoch 66 train≈val≈0.95 即此症状)。
    # 小样本诊断 (--subset 2000 → train_rmse 0.026) 已证明架构健康, 故改用 cosine 全程退火。
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup),
                                                     eta_min=args.min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[args.warmup])
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
start_epoch, best_val = 0, float('inf')

if args.resume and os.path.exists(os.path.join(SCRIPT_DIR, 'dimenetpp_last.pt')):
    ck = torch.load(os.path.join(SCRIPT_DIR, 'dimenetpp_last.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    start_epoch = ck['epoch'] + 1
    best_val = ck['best_val']
    print(f'[resume] 从 epoch {start_epoch} 继续 (best_val={best_val:.5f})')

# ===== 评估 =====
@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds, ys = [], []
    for data in loader:
        data = data.to(device)
        preds.append(model(data.z, data.pos, data.batch))
        ys.append(data.y)
    return torch.cat(preds), torch.cat(ys)

# ===== 训练 =====
log_path = os.path.join(SCRIPT_DIR, 'dimenet_train_log.csv')
with open(log_path, 'a' if args.resume else 'w', newline='') as f:
    w = csv.writer(f)
    if not args.resume:
        w.writerow(['epoch', 'lr', 'train_rmse', 'val_rmse', 'best_val', 'sec'])

patience_left = args.patience if args.schedule == 'plateau' else args.epochs  # cosine 不早停, 跑满
print(f'开始训练: {args.epochs} epochs, batch {args.batch_size}, workers {args.workers}', flush=True)
for epoch in range(start_epoch, args.epochs):
    t_ep = time.time()
    model.train()
    total, n = 0.0, 0
    for bi, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.z, data.pos, data.batch)
        loss = F.mse_loss(pred, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * data.num_graphs
        n += data.num_graphs
        if (bi + 1) % 500 == 0:
            print(f'  epoch {epoch+1}: {bi+1}/{len(train_loader)} 步, 已用 {time.time()-t_ep:.0f}s, loss {loss.item():.4f}', flush=True)
    train_rmse = (total / n) ** 0.5

    if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
        val_pred, val_y = evaluate(val_loader)
        val_rmse = float(F.mse_loss(val_pred, val_y).sqrt())
    else:
        val_rmse = float('nan')
    if args.schedule == 'cosine':
        scheduler.step()
    elif not np.isnan(val_rmse):
        scheduler.step(val_rmse)
    lr = optimizer.param_groups[0]['lr']

    improved = val_rmse < best_val - 1e-6
    if improved:
        best_val = val_rmse
        patience_left = args.patience
        torch.save({'model': model.state_dict(), 'best_val': best_val, 'epoch': epoch},
                   os.path.join(SCRIPT_DIR, 'dimenetpp_best.pt'))
    elif args.schedule == 'plateau':
        patience_left -= 1
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'best_val': best_val, 'epoch': epoch},
               os.path.join(SCRIPT_DIR, 'dimenetpp_last.pt'))

    sec = time.time() - t_ep
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([epoch + 1, f'{lr:.2e}', f'{train_rmse:.5f}', f'{val_rmse:.5f}',
                                f'{best_val:.5f}', f'{sec:.0f}'])
    print(f'Epoch {epoch+1:3d}/{args.epochs}  lr={lr:.1e}  train_rmse={train_rmse:.4f}  '
          f'val_rmse={val_rmse:.4f}  best={best_val:.4f}  {sec:.0f}s/epoch', flush=True)
    if patience_left <= 0:
        print(f'早停: val 连续 {args.patience} epoch 无改善')
        break

# ===== 测试集评估 (反标准化, 逐性质 R²) =====
print('=' * 70)
print('测试集评估 (13,547, 反标准化)')
print('=' * 70)
model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'dimenetpp_best.pt'), map_location=device, weights_only=False)['model'])
test_pred, test_y = evaluate(test_loader)
pred_unstd = test_pred.cpu().numpy() * y_std.cpu().numpy() + y_mean.cpu().numpy()
true_unstd = test_y.cpu().numpy() * y_std.cpu().numpy() + y_mean.cpu().numpy()

from sklearn.metrics import r2_score, mean_absolute_error
r2s, maes = [], []
print(f'{"property":<15} {"R2":>8} {"MAE":>10}')
for i, t in enumerate(TARGETS):
    r2 = r2_score(true_unstd[:, i], pred_unstd[:, i])
    mae = mean_absolute_error(true_unstd[:, i], pred_unstd[:, i])
    r2s.append(r2); maes.append(mae)
    print(f'{t:<15} {r2:>8.4f} {mae:>10.4f}')
print(f'{"mean":<15} {np.mean(r2s):>8.4f} {np.mean(maes):>10.4f}')
np.savez(os.path.join(SCRIPT_DIR, 'dimenet_test_preds.npz'),
         y_true=true_unstd, y_pred=pred_unstd, targets=np.array(TARGETS))
print('saved: dimenet_test_preds.npz')
print(f'[verify] mean R2 = {np.mean(r2s):.4f}   (对照: 3D GAT 集成 0.903 / SMILES GAT 0.932, 同一 13,547 测试集)')
