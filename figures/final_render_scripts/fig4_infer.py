# -*- coding: utf-8 -*-
"""Fig 4 (服务器版): Physics consistency (HL-Gap = LUMO - HOMO) — 推理 + 画图一体
采样与 physics_consistency.ipynb 完全一致 (seed(42) 随机 5000, SMILES 图补零对齐 XYZ 输入)。

⚠️ SMILES 面板定稿数字 (0.20 mHa, 99.7% <0.001 Ha) 来自权重 models/tmqm_gat_best_backup.pt
   (7月20日版), 不是 tmqm_gat_best.pt (7月15日版跑出 0.14 mHa)。勿换权重!

用法:
  python fig4_infer.py --xyz      只跑 3D GAT 推理 → 存 fig4_xyz_results.npz
  python fig4_infer.py --smiles   只跑 SMILES GAT 推理 → 存 fig4_smiles_results.npz
  python fig4_infer.py            两个 npz 都在 → 直接画图; 缺 → 先补跑推理再画图
  python fig4_infer.py --ckpt-file=models/xxx.pt   换 SMILES 权重 (默认 backup)
纯画图 (不需要 torch, 只调布局用): python fig4_plot.py

图注 P79 锚点: (a) R²=0.999 (b) mean=1.25 mHa (c) mean=0.20 mHa
(d) violation<1% (e) 3D/SMILES/RGCN 对比 (f) deviation vs true gap
输出: fig4_physics_hd.png"""
import numpy as np, pandas as pd, torch, torch.nn as nn, os, sys, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows GBK 控制台兜底; Linux 无害
except Exception:
    pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 布局参数 (notebook 大画布风格) ============
DPI = 600
FIGW, FIGH = 22.0, 15.0        # 2×3 六面板, 每面板约 7.3in 宽
FS_BASE, FS_TITLE, FS_AXLABEL = 13, 13, 13
FS_TICK, FS_LEGEND, FS_LEGEND_E, FS_SUPTITLE = 12, 12, 11, 16
# ======================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from tqdm import tqdm

HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
NPZ_XYZ = 'fig4_xyz_results.npz'
NPZ_SM = 'fig4_smiles_results.npz'

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
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0: attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, N)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(N, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out)

def collate_graphs(graph_list):
    x_list, ei_list, ea_list, batch_list = [], [], [], []
    offset = 0
    for i, g in enumerate(graph_list):
        n = g['x'].size(0)
        x_list.append(g['x']); ei_list.append(g['edge_index'] + offset)
        ea_list.append(g['edge_attr'])
        batch_list.append(torch.full((n,), i, dtype=torch.long, device=g['x'].device))
        offset += n
    return {'x': torch.cat(x_list, dim=0), 'edge_index': torch.cat(ei_list, dim=1), 'edge_attr': torch.cat(ea_list, dim=0)}, torch.cat(batch_list, dim=0)

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

class XYZ_GNN_Eval(nn.Module):
    def __init__(self, node_dim=11, edge_dim=23, hidden=HIDDEN, n_layers=N_LAYERS, output_dim=8, heads=GAT_HEADS):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden // heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(0.10)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden // 2, output_dim))
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

class SMILES_GAT_Eval(nn.Module):
    def __init__(self, node_dim=8, edge_dim=6, hidden=HIDDEN, n_layers=N_LAYERS, output_dim=8, heads=GAT_HEADS):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden // heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(0.10)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden // 2, output_dim))
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

# ===== 数据 (与 physics_consistency.ipynb Cell 3 相同) =====
df_y = pd.read_csv('models/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'})

ELEMENT_DATA = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES = [1, 2, 3, 12]; B2I = {b: i for i, b in enumerate(BOND_TYPES)}; N_BT = len(BOND_TYPES)

def atom_features(atom):
    z = atom.GetAtomicNum(); p, g = ELEMENT_DATA.get(z, (0, 0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g,
            int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    N = x.size(0); src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za >= 21) != (zb >= 21)); is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * N_BT
        if bt in B2I: bf[B2I[bt]] = 1.0
        bf.extend([is_ring, is_ml]); battr.extend([bf, bf])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea_smiles = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, N_BT + 2)
    x_xyz = torch.cat([x, torch.zeros(N, 3)], dim=1)
    ea_xyz = torch.cat([ea_smiles, torch.zeros(ea_smiles.size(0), 17)], dim=1)
    return {'x_smiles': x, 'edge_attr_smiles': ea_smiles,
            'x': x_xyz, 'edge_attr': ea_xyz,
            'edge_index': ei, 'num_nodes': N}

# ===== 采样 (seed 42, N=5000 — 每次运行结果一致, 分阶段跑两次可对齐) =====
np.random.seed(42)
smiles_list = [s for s in df_y['SMILES'].tolist() if isinstance(s, str) and s != '']
sample_idx = np.random.choice(len(smiles_list), min(5000, len(smiles_list)), replace=False)
print(f'Sampling {len(sample_idx)} complexes...')

graphs, y_true, valid_idx = [], [], []
for idx in tqdm(sample_idx, desc='Building graphs'):
    smi = smiles_list[idx]
    g = smiles_to_graph(smi)
    if g is None: continue
    graphs.append(g)
    y_true.append(df_y.iloc[idx][targets].values.astype(np.float32))
    valid_idx.append(idx)
y_true = np.array(y_true)
print(f'Built {len(graphs)} graphs')

target_cols = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
y_mean = df_y[target_cols].mean().values.astype(np.float32)
y_std = df_y[target_cols].std().values.astype(np.float32)

def run_inference(model, graphs, key_x, key_ea):
    bs = 128
    preds_list = []
    with torch.no_grad():
        for s in tqdm(range(0, len(graphs), bs), desc='inference'):
            bg = [{'x': g[key_x].to(device), 'edge_index': g['edge_index'].to(device),
                   'edge_attr': g[key_ea].to(device), 'num_nodes': g['num_nodes']}
                  for g in graphs[s:s + bs]]
            batched, batch_idx = collate_graphs(bg)
            preds_list.append(model(batched, batch_idx).cpu().numpy())
    preds_norm = np.concatenate(preds_list, axis=0)
    return preds_norm * y_std + y_mean

def infer_xyz():
    model_xyz = XYZ_GNN_Eval().to(device)
    state = torch.load('models/xyz_best.pt', map_location=device)
    model_xyz.load_state_dict(state, strict=False)
    print('Loaded models/xyz_best.pt')
    preds_xyz = run_inference(model_xyz, graphs, 'x', 'edge_attr')
    gap_pred, homo_pred, lumo_pred = preds_xyz[:, 0], preds_xyz[:, 1], preds_xyz[:, 2]
    gap_derived = lumo_pred - homo_pred
    deviation = np.abs(gap_pred - gap_derived)
    print(f'XYZ GNN: mean={deviation.mean()*1000:.3f} mHa  compliance<0.01={(deviation<0.01).mean()*100:.1f}%')
    np.savez_compressed(NPZ_XYZ,
                        gap_pred=gap_pred, gap_derived=gap_derived, deviation=deviation,
                        homo_pred=homo_pred, lumo_pred=lumo_pred, gap_true=y_true[:, 0])
    print(f'saved {NPZ_XYZ}')

def infer_smiles():
    sm_ckpt = None
    for a in sys.argv[1:]:
        if a.startswith('--ckpt-file='):
            sm_ckpt = a.split('=', 1)[1]
    if sm_ckpt is None:
        sm_ckpt = 'models/tmqm_best.pt' if '--sm-old' in sys.argv else 'models/tmqm_gat_best_backup.pt'
        # Fig4 定稿数字 0.20 mHa / 99.7% <0.001 Ha 来自 backup 权重, 勿换
    model_smiles = SMILES_GAT_Eval().to(device)
    state = torch.load(sm_ckpt, map_location=device)
    model_smiles.load_state_dict(state, strict=False)
    print(f'Loaded {sm_ckpt}')
    preds_sm = run_inference(model_smiles, graphs, 'x_smiles', 'edge_attr_smiles')
    gap_pred_sm, lumo_pred_sm, homo_pred_sm = preds_sm[:, 0], preds_sm[:, 2], preds_sm[:, 1]
    gap_derived_sm = lumo_pred_sm - homo_pred_sm
    deviation_sm = np.abs(gap_pred_sm - gap_derived_sm)
    print(f'SMILES GAT: mean={deviation_sm.mean()*1000:.3f} mHa  compliance<0.01={(deviation_sm<0.01).mean()*100:.1f}%')
    print(f'SMILES GAT: compliance<0.001={(deviation_sm<0.001).mean()*100:.1f}% (P76 定稿: 99.7%)')
    np.savez_compressed(NPZ_SM,
                        gap_pred=gap_pred_sm, gap_derived=gap_derived_sm,
                        deviation=deviation_sm)
    print(f'saved {NPZ_SM}')

# ===== 运行模式 =====
if '--xyz' in sys.argv:
    infer_xyz(); sys.exit(0)
if '--smiles' in sys.argv:
    infer_smiles(); sys.exit(0)
have_npz = os.path.exists(NPZ_XYZ) and os.path.exists(NPZ_SM)
if not have_npz:
    if not os.path.exists(NPZ_XYZ): infer_xyz()
    if not os.path.exists(NPZ_SM): infer_smiles()

xz = np.load(NPZ_XYZ); sm = np.load(NPZ_SM)
xyz_results = {k: xz[k] for k in xz.files}
smiles_results = {k: sm[k] for k in sm.files}
print('npz loaded')

# ===== 高清画图: 2×3 =====
plt.rcParams.update({'font.size': FS_BASE, 'axes.titlesize': FS_TITLE, 'axes.labelsize': FS_AXLABEL,
                     'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEGEND})
fig, axes = plt.subplots(2, 3, figsize=(FIGW, FIGH))

ax = axes[0, 0]
ax.scatter(xyz_results['gap_derived'], xyz_results['gap_pred'], alpha=0.3, s=6, c='darkorange')
lims = [min(xyz_results['gap_pred'].min(), xyz_results['gap_derived'].min()),
        max(xyz_results['gap_pred'].max(), xyz_results['gap_derived'].max())]
ax.plot(lims, lims, 'r--', lw=1)
r2_xyz_gap = np.corrcoef(xyz_results['gap_derived'], xyz_results['gap_pred'])[0, 1] ** 2
ax.set_title(f'(a) 3D GAT: gap vs (LUMO−HOMO)\nR² = {r2_xyz_gap:.4f}')
ax.set_xlabel('LUMO_pred − HOMO_pred (Ha)'); ax.set_ylabel('gap_pred (Ha)')

ax = axes[0, 1]
ax.hist(xyz_results['deviation'] * 1000, bins=80, color='darkorange', edgecolor='white', alpha=0.8)
ax.axvline(x=xyz_results['deviation'].mean() * 1000, color='red', linestyle='--',
           label=f'Mean = {xyz_results["deviation"].mean()*1000:.2f} mHa')
ax.set_title('(b) 3D GAT: |gap − (LUMO−HOMO)| distribution')
ax.set_xlabel('Deviation (mHa)'); ax.set_ylabel('Count'); ax.legend()

ax = axes[0, 2]
ax.scatter(smiles_results['gap_derived'], smiles_results['gap_pred'], alpha=0.3, s=6, c='steelblue')
lims = [min(smiles_results['gap_pred'].min(), smiles_results['gap_derived'].min()),
        max(smiles_results['gap_pred'].max(), smiles_results['gap_derived'].max())]
ax.plot(lims, lims, 'r--', lw=1)
r2_sm_gap = np.corrcoef(smiles_results['gap_derived'], smiles_results['gap_pred'])[0, 1] ** 2
ax.set_title(f'(c) SMILES GAT: gap vs (LUMO−HOMO)\nR² = {r2_sm_gap:.4f}')
ax.set_xlabel('LUMO_pred − HOMO_pred (Ha)'); ax.set_ylabel('gap_pred (Ha)')

ax = axes[1, 0]
homo_ok = xyz_results['homo_pred'] < xyz_results['lumo_pred']
violation_rate = (~homo_ok).mean() * 100
colors = ['#2ecc71' if ok else '#e74c3c' for ok in homo_ok]
ax.scatter(xyz_results['homo_pred'][:500], xyz_results['lumo_pred'][:500],
           c=colors[:500], alpha=0.5, s=9)
ax.plot([-0.5, 0.1], [-0.5, 0.1], 'k--', lw=0.5)
ax.set_xlabel('HOMO_pred (Ha)'); ax.set_ylabel('LUMO_pred (Ha)')
ax.set_title(f'(d) 3D GAT: HOMO < LUMO ordering\nViolation rate: {violation_rate:.2f}% (green = OK, red = violation)')

ax = axes[1, 1]
methods = ['3D GAT', 'SMILES GAT', 'Published\nRGCN']
mean_devs = [xyz_results['deviation'].mean() * 1000, smiles_results['deviation'].mean() * 1000, 0.1]
comply_010 = [(xyz_results['deviation'] < 0.010).mean() * 100,
              (smiles_results['deviation'] < 0.010).mean() * 100, 100]
x = np.arange(len(methods)); w = 0.25
ax.bar(x - w, mean_devs, w, label='Mean deviation (mHa)', color='steelblue')
ax2 = ax.twinx()
ax2.bar(x + w, comply_010, w, label='Compliance <0.01 Ha (%)', color='darkorange')
ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel('Mean |gap − (LUMO−HOMO)| (mHa)')
ax2.set_ylabel('Compliance <0.01 Ha (%)'); ax.set_title('(e) Physics Consistency Comparison')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
          bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=FS_LEGEND_E)

ax = axes[1, 2]
ax.scatter(xyz_results['gap_true'], xyz_results['deviation'] * 1000, alpha=0.2, s=6, c='darkorange')
ax.set_xlabel('True gap (Ha)'); ax.set_ylabel('|gap_pred − (LUMO−HOMO)| (mHa)')
ax.set_title('(f) 3D GAT: Deviation vs True Gap')
ax.axhline(y=xyz_results['deviation'].mean() * 1000, color='red', linestyle='--', alpha=0.7)

fig.suptitle('Physics Consistency: HL-Gap = LUMO − HOMO Verification', fontsize=FS_SUPTITLE, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0.09, 1, 1])
out = 'fig4_physics_hd.png'
plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f'fig4 saved: {out}')

# 验证数字 vs 图注 P79
print(f'[verify] (a) R²={r2_xyz_gap:.4f} (caption: 0.999)')
print(f'[verify] (b) mean={xyz_results["deviation"].mean()*1000:.3f} mHa (caption: 1.25)')
print(f'[verify] (c) mean={smiles_results["deviation"].mean()*1000:.3f} mHa (caption: 0.20)')
print(f'[verify] (d) violation={violation_rate:.3f}% (caption: <1%)')
from PIL import Image
im = Image.open(out)
print(f'[size] {im.size[0]}x{im.size[1]}, extent cy = {round(5486400 * im.size[1] / im.size[0])}')
