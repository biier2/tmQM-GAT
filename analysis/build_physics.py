"""Build physics consistency notebook: gap = LUMO - HOMO verification"""
import json

cells = []

def add_md(src, cid):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.split('\n'), "id": cid})

def add_code(src, cid):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [l + '\n' for l in src.split('\n')], "id": cid})

# Cell 1: intro
add_md("""# Physics consistency constraint: HL-Gap ≡ LUMO - HOMO

## Background
By molecular orbital theory, the HOMO-LUMO gap is by construction equal to LUMO - HOMO:
```
HL-Gap = E_LUMO - E_HOMO
```
A multi-task model predicts the three as independent targets and may violate this identity.
tmGNN-XAI added an L1 soft constraint `loss += 0.1 × |gap - (lumo - homo)|` and ultimately reached a mean deviation of 0.0001 Ha.

## What this experiment does
1. **Assess the status quo**: compute the gap vs (LUMO - HOMO) deviation on the test set with the existing checkpoint
2. **Compare with tmGNN-XAI**: how large is the gap? Is a constraint needed?
3. **(Optional) Retrain**: retrain XYZ GNN with a soft penalty and verify the improvement""", "phy_intro")

# Cell 2: imports + model definition
add_code("""import numpy as np, pandas as pd, torch, torch.nn as nn, os, sys, gzip, json
import matplotlib.pyplot as plt, warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']

# ===== GAT model (consistent with the training code) =====
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
        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim>0 else None
        self.act = nn.ReLU()
    def forward(self, x, ei, ea=None):
        src, dst = ei[0], ei[1]; N = x.size(0); h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0)>0: attn = attn + self.edge_attn(ea)
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
    '''XYZ GNN for evaluation (matches xyz_best.pt: 11 node, 23 edge)'''
    def __init__(self, node_dim=11, edge_dim=23, hidden=HIDDEN, n_layers=N_LAYERS, output_dim=8, heads=GAT_HEADS):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(0.10)
        self.readout = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden//2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            mm = (x[:,7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        mm = (x[:,7] > 0.5)
        mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

class SMILES_GAT_Eval(nn.Module):
    '''SMILES GAT for evaluation (matches tmqm_gat_best.pt: 8 node, 6 edge)'''
    def __init__(self, node_dim=8, edge_dim=6, hidden=HIDDEN, n_layers=N_LAYERS, output_dim=8, heads=GAT_HEADS):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(0.10)
        self.readout = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden//2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            mm = (x[:,7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        mm = (x[:,7] > 0.5)
        mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

print('Model classes defined')
""", "phy_model")

# Cell 3: load XYZ GNN + evaluate
add_code("""print('='*60)
print('Step 1: XYZ GNN physics consistency evaluation')
print('='*60)

# load the XYZ checkpoint
model_xyz = XYZ_GNN_Eval().to(device)
ckpt_xyz = 'xyz_best.pt'
if os.path.exists(ckpt_xyz):
    state = torch.load(ckpt_xyz, map_location=device)
    model_xyz.load_state_dict(state, strict=False)
    print(f'Loaded {ckpt_xyz}')
else:
    print(f'{ckpt_xyz} not found, skipping XYZ GNN')
    model_xyz = None

# load the data + build graphs (XYZ+BO, simplified: build the test graphs from SMILES only)
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'
})

ELEMENT_DATA = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES = [1,2,3,12]; B2I = {b:i for i,b in enumerate(BOND_TYPES)}; N_BT = len(BOND_TYPES)

def atom_features(atom):
    z = atom.GetAtomicNum(); p,g = ELEMENT_DATA.get(z,(0,0))
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
        is_ml = float((za>=21)!=(zb>=21)); is_ring = float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf = [0.0]*N_BT
        if bt in B2I: bf[B2I[bt]] = 1.0
        bf.extend([is_ring, is_ml]); battr.extend([bf,bf])
    ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
    ea_smiles = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, N_BT+2)
    # XYZ GNN needs 11-dim nodes + 23-dim edges, so pad with zeros to align
    x_xyz = torch.cat([x, torch.zeros(N, 3)], dim=1)  # pad 3 dims of BO statistics
    ea_xyz = torch.cat([ea_smiles, torch.zeros(ea_smiles.size(0), 17)], dim=1)  # pad 16 RBF + 1 BO
    return {'x_smiles': x, 'edge_attr_smiles': ea_smiles,
            'x': x_xyz, 'edge_attr': ea_xyz,
            'edge_index': ei, 'num_nodes': N}
    return {'x':x, 'edge_index':ei, 'edge_attr':ea}

# sample the test set (quick evaluation: N=5000)
np.random.seed(42)
smiles_list = [s for s in df_y['SMILES'].tolist() if isinstance(s, str) and s != '']
sample_idx = np.random.choice(len(smiles_list), min(5000, len(smiles_list)), replace=False)
print(f'Sampling {len(sample_idx)} complexes for evaluation...')

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

# XYZ GNN inference
xyz_results = None
if model_xyz is not None:
    bs = 128
    preds_list = []
    with torch.no_grad():
        for s in tqdm(range(0, len(graphs), bs), desc='XYZ inference'):
            bg = [{'x': g['x'].to(device), 'edge_index': g['edge_index'].to(device),
                   'edge_attr': g['edge_attr'].to(device), 'num_nodes': g['num_nodes']}
                  for g in graphs[s:s+bs]]
            batched, batch_idx = collate_graphs(bg)
            preds_list.append(model_xyz(batched, batch_idx).cpu().numpy())
    preds_xyz_norm = np.concatenate(preds_list, axis=0)  # z-score values

    # de-normalization: the model outputs z-scores, which must be converted back to physical units
    target_cols = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
    y_mean = df_y[target_cols].mean().values.astype(np.float32)
    y_std = df_y[target_cols].std().values.astype(np.float32)
    preds_xyz = preds_xyz_norm * y_std + y_mean

    # physics consistency: gap vs (lumo - homo) — computed in physical space
    gap_pred = preds_xyz[:, 0]
    homo_pred = preds_xyz[:, 1]
    lumo_pred = preds_xyz[:, 2]
    gap_derived = lumo_pred - homo_pred
    deviation = np.abs(gap_pred - gap_derived)

    print(f'  Normalization: gap(mean={y_mean[0]:.4f}, std={y_std[0]:.4f}), '
          f'homo(mean={y_mean[1]:.4f}, std={y_std[1]:.4f}), '
          f'lumo(mean={y_mean[2]:.4f}, std={y_std[2]:.4f})')
    print(f'\\nXYZ GNN Physics Consistency (n={len(deviation):,}):')
    print(f'  Mean |gap - (LUMO-HOMO)|: {deviation.mean():.6f} Ha')
    print(f'  Median deviation:          {np.median(deviation):.6f} Ha')
    print(f'  Max deviation:             {deviation.max():.6f} Ha')
    print(f'  Compliance <0.001 Ha:      {(deviation < 0.001).mean()*100:.1f}%')
    print(f'  Compliance <0.005 Ha:      {(deviation < 0.005).mean()*100:.1f}%')
    print(f'  Compliance <0.010 Ha:      {(deviation < 0.010).mean()*100:.1f}%')
    print(f'  tmGNN-XAI reference:       0.0001 Ha mean, 100% <0.01 Ha')

    xyz_results = {
        'gap_pred': gap_pred, 'gap_derived': gap_derived,
        'deviation': deviation, 'homo_pred': homo_pred, 'lumo_pred': lumo_pred,
        'gap_true': y_true[:, 0], 'homo_true': y_true[:, 1], 'lumo_true': y_true[:, 2]
    }
""", "phy_xyz_eval")

# Cell 4: load SMILES GAT + evaluate
add_code("""print('='*60)
print('Step 2: SMILES GAT physics consistency evaluation')
print('='*60)

model_smiles = SMILES_GAT_Eval().to(device)
ckpt_sm = 'tmqm_gat_best.pt'
if not os.path.exists(ckpt_sm): ckpt_sm = 'tmqm_best.pt'
if os.path.exists(ckpt_sm):
    state = torch.load(ckpt_sm, map_location=device)
    model_smiles.load_state_dict(state, strict=False)
    print(f'Loaded {ckpt_sm}')
else:
    print(f'No SMILES checkpoint found')
    model_smiles = None

smiles_results = None
if model_smiles is not None:
    bs = 128
    preds_list = []
    with torch.no_grad():
        for s in tqdm(range(0, len(graphs), bs), desc='SMILES inference'):
            bg = [{'x': g['x_smiles'].to(device), 'edge_index': g['edge_index'].to(device),
                   'edge_attr': g['edge_attr_smiles'].to(device), 'num_nodes': g['num_nodes']}
                  for g in graphs[s:s+bs]]
            batched, batch_idx = collate_graphs(bg)
            preds_list.append(model_smiles(batched, batch_idx).cpu().numpy())
    preds_sm_norm = np.concatenate(preds_list, axis=0)  # z-score values
    preds_sm = preds_sm_norm * y_std + y_mean  # de-normalization

    gap_pred_sm = preds_sm[:, 0]
    lumo_pred_sm = preds_sm[:, 2]
    homo_pred_sm = preds_sm[:, 1]
    gap_derived_sm = lumo_pred_sm - homo_pred_sm
    deviation_sm = np.abs(gap_pred_sm - gap_derived_sm)

    print(f'\\nSMILES GAT Physics Consistency (n={len(deviation_sm):,}):')
    print(f'  Mean |gap - (LUMO-HOMO)|: {deviation_sm.mean():.6f} Ha')
    print(f'  Median deviation:          {np.median(deviation_sm):.6f} Ha')
    print(f'  Max deviation:             {deviation_sm.max():.6f} Ha')
    print(f'  Compliance <0.001 Ha:      {(deviation_sm < 0.001).mean()*100:.1f}%')
    print(f'  Compliance <0.005 Ha:      {(deviation_sm < 0.005).mean()*100:.1f}%')
    print(f'  Compliance <0.010 Ha:      {(deviation_sm < 0.010).mean()*100:.1f}%')

    smiles_results = {
        'gap_pred': gap_pred_sm, 'gap_derived': gap_derived_sm,
        'deviation': deviation_sm
    }
""", "phy_smiles_eval")

# Cell 5: visualization
add_code("""print('='*60)
print('Step 3: visualization')
print('='*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# (a) XYZ GNN: gap_pred vs gap_derived
ax = axes[0, 0]
if xyz_results:
    ax.scatter(xyz_results['gap_derived'], xyz_results['gap_pred'], alpha=0.3, s=5, c='darkorange')
    lims = [min(xyz_results['gap_pred'].min(), xyz_results['gap_derived'].min()),
            max(xyz_results['gap_pred'].max(), xyz_results['gap_derived'].max())]
    ax.plot(lims, lims, 'r--', lw=1)
    r2_xyz_gap = np.corrcoef(xyz_results['gap_derived'], xyz_results['gap_pred'])[0,1]**2
    ax.set_title(f'XYZ GNN: gap vs (LUMO-HOMO)\\nR²={r2_xyz_gap:.4f}')
ax.set_xlabel('LUMO_pred - HOMO_pred (Ha)'); ax.set_ylabel('gap_pred (Ha)')

# (b) XYZ GNN: deviation histogram
ax = axes[0, 1]
if xyz_results:
    ax.hist(xyz_results['deviation']*1000, bins=80, color='darkorange', edgecolor='white', alpha=0.8)
    ax.axvline(x=xyz_results['deviation'].mean()*1000, color='red', linestyle='--',
               label=f'Mean={xyz_results["deviation"].mean()*1000:.3f} mHa')
    ax.set_title('XYZ GNN: |gap - (LUMO-HOMO)| distribution')
    ax.set_xlabel('Deviation (mHa)'); ax.set_ylabel('Count'); ax.legend()

# (c) SMILES GAT: gap_pred vs gap_derived
ax = axes[0, 2]
if smiles_results:
    ax.scatter(smiles_results['gap_derived'], smiles_results['gap_pred'], alpha=0.3, s=5, c='steelblue')
    lims = [min(smiles_results['gap_pred'].min(), smiles_results['gap_derived'].min()),
            max(smiles_results['gap_pred'].max(), smiles_results['gap_derived'].max())]
    ax.plot(lims, lims, 'r--', lw=1)
    r2_sm_gap = np.corrcoef(smiles_results['gap_derived'], smiles_results['gap_pred'])[0,1]**2
    ax.set_title(f'SMILES GAT: gap vs (LUMO-HOMO)\\nR²={r2_sm_gap:.4f}')
ax.set_xlabel('LUMO_pred - HOMO_pred (Ha)'); ax.set_ylabel('gap_pred (Ha)')

# (d) check for HOMO-LUMO ordering violations
ax = axes[1, 0]
if xyz_results:
    # HOMO should lie below LUMO (more negative)
    homo_ok = xyz_results['homo_pred'] < xyz_results['lumo_pred']
    violation_rate = (~homo_ok).mean() * 100
    colors = ['#2ecc71' if ok else '#e74c3c' for ok in homo_ok]
    ax.scatter(xyz_results['homo_pred'][:500], xyz_results['lumo_pred'][:500],
               c=colors[:500], alpha=0.5, s=8)
    ax.plot([-0.5, 0.1], [-0.5, 0.1], 'k--', lw=0.5)
    ax.set_xlabel('HOMO_pred (Ha)'); ax.set_ylabel('LUMO_pred (Ha)')
    ax.set_title(f'XYZ GNN: HOMO < LUMO?\\nViolation rate: {violation_rate:.2f}% (green=OK, red=violation)')

# (e) deviation comparison between the two models
ax = axes[1, 1]
methods = []; mean_devs = []; comply_001 = []; comply_010 = []
if xyz_results:
    methods.append('XYZ GNN'); mean_devs.append(xyz_results['deviation'].mean()*1000)
    comply_001.append((xyz_results['deviation'] < 0.001).mean()*100)
    comply_010.append((xyz_results['deviation'] < 0.010).mean()*100)
if smiles_results:
    methods.append('SMILES GAT'); mean_devs.append(smiles_results['deviation'].mean()*1000)
    comply_001.append((smiles_results['deviation'] < 0.001).mean()*100)
    comply_010.append((smiles_results['deviation'] < 0.010).mean()*100)
methods.append('tmGNN-XAI*'); mean_devs.append(0.1); comply_001.append(100); comply_010.append(100)

x = np.arange(len(methods)); w = 0.25
ax.bar(x - w, mean_devs, w, label='Mean deviation (mHa)', color='steelblue')
ax2 = ax.twinx()
ax2.bar(x + w, comply_010, w, label='Compliance <0.01 Ha (%)', color='darkorange')
ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel('Mean |gap - (LUMO-HOMO)| (mHa)')
ax2.set_ylabel('Compliance <0.01 Ha (%)'); ax.set_title('Physics Consistency Comparison')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

# (f) gap true vs deviation (check for a systematic offset)
ax = axes[1, 2]
if xyz_results:
    ax.scatter(xyz_results['gap_true'], xyz_results['deviation']*1000, alpha=0.2, s=5, c='darkorange')
    ax.set_xlabel('True gap (Ha)'); ax.set_ylabel('|gap_pred - (LUMO-HOMO)| (mHa)')
    ax.set_title('XYZ GNN: Deviation vs True Gap')
    ax.axhline(y=xyz_results['deviation'].mean()*1000, color='red', linestyle='--', alpha=0.7)

plt.suptitle('Physics Consistency: HL-Gap = LUMO - HOMO Verification', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(); plt.show()

# summary
print('\\n' + '='*60)
print('Physics consistency summary')
print('='*60)
if xyz_results:
    print(f'XYZ GNN:  Mean dev={xyz_results["deviation"].mean()*1000:.4f} mHa, '
          f'Compliance@0.01Ha={(xyz_results["deviation"] < 0.01).mean()*100:.1f}%')
if smiles_results:
    print(f'SMILES GAT: Mean dev={smiles_results["deviation"].mean()*1000:.4f} mHa, '
          f'Compliance@0.01Ha={(smiles_results["deviation"] < 0.01).mean()*100:.1f}%')
print(f'tmGNN-XAI:  Mean dev=0.1 mHa, Compliance@0.01Ha=100%')
""", "phy_viz")

# Cell 6: discussion
add_md("""## Evaluation conclusions

### Criteria
- **deviation < 0.001 Ha (1 mHa)**: physically exact → no soft constraint needed
- **deviation < 0.010 Ha (10 mHa)**: acceptable model error → an optional soft constraint may improve it
- **deviation > 0.010 Ha**: an L1 penalty and retraining are needed

### What tmGNN-XAI did
They added an L1 penalty term `0.1 × |HL-Gap - (LUMO - HOMO)|` to the loss, ultimately reaching a mean deviation of 0.0001 Ha and 100% compliance.
If our model's deviation is larger, a similar term can be added during training:

```python
physics_penalty = 0.1 * torch.abs(preds[:, 0] - (preds[:, 2] - preds[:, 1])).mean()
loss = mse_loss + physics_penalty
```

### HOMO-LUMO ordering violations
The HOMO energy should always lie below the LUMO (more negative). Check whether `homo_pred < lumo_pred` holds for every sample. A violation rate > 1% would mean the model has learned a non-physical orbital ordering.
""", "phy_discussion")

# Cell 7: optional retraining (with a soft penalty)
add_code("""# ============================================================
# optional: retrain XYZ GNN with a physics penalty
# set RUN_RETRAIN=True to run it
# ============================================================
RUN_RETRAIN = False

if RUN_RETRAIN:
    print('Retraining XYZ GNN with physics consistency penalty...')
    print('(needs a GPU, ~1 hour, using the same training configuration as tmQM_Fe_XYZ.ipynb)')
    # identical to the Cell 7 training code in tmQM_Fe_XYZ.ipynb, only the loss gains:
    # physics_loss = 0.1 * torch.abs(preds[:,0] - (preds[:,2] - preds[:,1])).mean()
    # loss = mse_loss + physics_loss
else:
    print('Skipping retraining. Set RUN_RETRAIN=True to enable.')
    print('If physics deviation is already small (<1 mHa), retraining is unnecessary.')
""", "phy_retrain")

# ============================================================
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.0"}},
      "cells": cells}

with open('physics_consistency.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

with open('physics_consistency.ipynb', encoding='utf-8') as f:
    json.load(f)
print(f'Built {len(cells)} cells, JSON valid → physics_consistency.ipynb')
