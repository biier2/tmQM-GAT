# -*- coding: utf-8 -*-
"""DimeNet++ 基线 — 仅测试集评估 (服务器 GPU, 约 2-3 分钟/检查点)

用途: R4 训练中途停止后, 对已保存的检查点补做正式的 13,547 测试集评估
(训练脚本只在跑完全部 epoch 后自动评估; 本脚本单独补这一步, 不训练、不动检查点)。

用法 (与 train_dimenet.py 同目录):
  python eval_dimenet.py --all         # 评估 best + last 两个检查点 (推荐)
  python eval_dimenet.py               # 只评估 dimenetpp_best.pt (协议检查点)
  python eval_dimenet.py --ckpt dimenetpp_last.pt   # 指定检查点

输出: 逐性质 R²/MAE + [verify] mean R2 对照行;
      best 检查点评估存 dimenet_test_preds.npz (与训练脚本最终产物同名),
      其他检查点存 dimenet_test_preds_<名>.npz。
"""
import os, sys, argparse
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
from sklearn.metrics import r2_score, mean_absolute_error

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=os.path.join(SCRIPT_DIR, 'dimenet_data.npz'))
parser.add_argument('--ckpt', default='dimenetpp_best.pt', help='检查点文件名 (--all 时忽略)')
parser.add_argument('--all', action='store_true', help='评估 best 与 last 两个检查点')
parser.add_argument('--batch-size', type=int, default=8)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

D = np.load(args.data, allow_pickle=True)
y_mean_cpu = torch.from_numpy(D['y_mean'].astype(np.float32))
y_std_cpu = torch.from_numpy(D['y_std'].astype(np.float32))

def make_dataset(sel, tag):
    """与 train_dimenet.py 完全一致 (NpzFile 列缓存 + z-score + (1,8) 形状)。"""
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
            y=((torch.from_numpy(y_arr[k]) - y_mean_cpu) / y_std_cpu).view(1, -1)))
    return data

test_ds = make_dataset(range(len(D['test_sel'])), 'test')
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
print(f'Test {len(test_ds)} (固定 13,547 测试集)', flush=True)

def build_model():
    """与 train_dimenet.py 的模型构造完全一致。"""
    return DimeNetPlusPlus(
        hidden_channels=128, out_channels=len(TARGETS), num_blocks=4,
        int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, cutoff=float(D['cutoff']),
        max_num_neighbors=int(D['max_neighbors']),
        envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3,
    ).to(device)

@torch.no_grad()
def evaluate(loader, model):
    model.eval()
    preds, ys = [], []
    for data in loader:
        data = data.to(device)
        preds.append(model(data.z, data.pos, data.batch))
        ys.append(data.y)
    return torch.cat(preds), torch.cat(ys)

def report(ckpt_name):
    path = os.path.join(SCRIPT_DIR, ckpt_name)
    assert os.path.exists(path), f'检查点不存在: {path}'
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model()
    model.load_state_dict(ck['model'])
    print('=' * 70, flush=True)
    print(f'检查点: {ckpt_name}  (epoch {ck.get("epoch", "?")}, best_val {ck.get("best_val", float("nan")):.5f})',
          flush=True)
    print('=' * 70, flush=True)
    test_pred, test_y = evaluate(test_loader, model)
    pred_unstd = test_pred.cpu().numpy() * y_std_cpu.numpy() + y_mean_cpu.numpy()
    true_unstd = test_y.cpu().numpy() * y_std_cpu.numpy() + y_mean_cpu.numpy()
    r2s, maes = [], []
    print(f'{"property":<15} {"R2":>8} {"MAE":>10}', flush=True)
    for i, t in enumerate(TARGETS):
        r2 = r2_score(true_unstd[:, i], pred_unstd[:, i])
        mae = mean_absolute_error(true_unstd[:, i], pred_unstd[:, i])
        r2s.append(r2); maes.append(mae)
        print(f'{t:<15} {r2:>8.4f} {mae:>10.4f}', flush=True)
    print(f'{"mean":<15} {np.mean(r2s):>8.4f} {np.mean(maes):>10.4f}', flush=True)
    print(f'[verify] {ckpt_name}: mean R2 = {np.mean(r2s):.4f}   '
          f'(对照: 3D GAT 集成 0.903 / SMILES GAT 0.932, 同一 13,547 测试集)', flush=True)
    out = ('dimenet_test_preds.npz' if ckpt_name == 'dimenetpp_best.pt'
           else f'dimenet_test_preds_{os.path.splitext(ckpt_name)[0]}.npz')
    np.savez(os.path.join(SCRIPT_DIR, out), y_true=true_unstd, y_pred=pred_unstd, targets=np.array(TARGETS))
    print(f'saved: {out}', flush=True)
    return np.mean(r2s)

if args.all:
    report('dimenetpp_best.pt')
    report('dimenetpp_last.pt')
else:
    report(args.ckpt)
