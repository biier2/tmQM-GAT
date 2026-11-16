# -*- coding: utf-8 -*-
"""DimeNet++ baseline — test-set-only evaluation (server GPU, about 2-3 minutes per checkpoint)

Purpose: after R4 training was stopped mid-way, run the official 13,547 test-set evaluation on the saved checkpoints
(the training script evaluates automatically only after all epochs finish; this script performs this step separately, without training or touching the checkpoints).

Usage (same directory as train_dimenet.py):
  python eval_dimenet.py --all         # evaluate both the best and last checkpoints (recommended)
  python eval_dimenet.py               # evaluate only dimenetpp_best.pt (protocol checkpoint)
  python eval_dimenet.py --ckpt dimenetpp_last.pt   # specify a checkpoint

Output: per-property R²/MAE + a [verify] mean R2 reference line;
      the best-checkpoint evaluation is saved as dimenet_test_preds.npz (same name as the training script's final product),
      other checkpoints are saved as dimenet_test_preds_<name>.npz.
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
parser.add_argument('--ckpt', default='dimenetpp_best.pt', help='checkpoint filename (ignored with --all)')
parser.add_argument('--all', action='store_true', help='evaluate both the best and last checkpoints')
parser.add_argument('--batch-size', type=int, default=8)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

D = np.load(args.data, allow_pickle=True)
y_mean_cpu = torch.from_numpy(D['y_mean'].astype(np.float32))
y_std_cpu = torch.from_numpy(D['y_std'].astype(np.float32))

def make_dataset(sel, tag):
    """Exactly identical to train_dimenet.py (NpzFile column cache + z-score + (1,8) shape)."""
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
print(f'Test {len(test_ds)} (fixed 13,547 test set)', flush=True)

def build_model():
    """Exactly identical to the model construction in train_dimenet.py."""
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
    assert os.path.exists(path), f'checkpoint does not exist: {path}'
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model()
    model.load_state_dict(ck['model'])
    print('=' * 70, flush=True)
    print(f'checkpoint: {ckpt_name}  (epoch {ck.get("epoch", "?")}, best_val {ck.get("best_val", float("nan")):.5f})',
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
          f'(reference: 3D GAT ensemble 0.903 / SMILES GAT 0.932, same 13,547 test set)', flush=True)
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
