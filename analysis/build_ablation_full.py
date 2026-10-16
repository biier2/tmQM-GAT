"""Build full-dataset ablation notebook for XYZ GNN key experiments"""
import json

cells = []

def add_code(src, cid):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [l + '\n' for l in src.split('\n')], "id": cid})

def add_md(src, cid):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.split('\n'), "id": cid})

add_md("""# XYZ GNN 消融实验 (全量95k, 5项)

| ID | 变量 | 对照 | 实验组 | 核心问题 |
|----|------|------|--------|---------|
| A | 金属池化 | on | off | 金属单独readout是否必要? |
| B | 注意力头数 | 8头 | 12头 | 多头是否过度参数化? |
| C | 金属加权 | 等权 | 反频率 | 稀有金属加权是否有效? |
| D | DFT边特征 | 23-dim | 6-dim (SMILES only) | 3D距离+BO对dipole的提升? |
| E | RBF分辨率 | 16 RBF (23-dim) | 8 RBF (15-dim) | 距离编码细度的影响? |
| F | Mayer BO消融 | node 11d+edge 23d | node 8d+edge 22d | BO独立贡献(保留16 RBF)? |

每实验约1h。5070 Ti 总计 ~6h。""", "eq_intro")

add_code("""import numpy as np, pandas as pd, time, json, os, sys, gzip
import torch, torch.nn as nn, torch.optim as optim
from torch.amp import autocast, GradScaler
from collections import Counter
import warnings; warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
from sklearn.metrics import r2_score
from tqdm import tqdm
from rdkit import Chem; from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

# ===== 数据加载 (复用tmQM_Fe_XYZ的Cell 2-5逻辑) =====
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={'HL_Gap':'gap','HOMO_Energy':'homo','LUMO_Energy':'lumo',
    'Dipole_M':'dipole','Metal_q':'metal_charge','Electronic_E':'E_elec',
    'Dispersion_E':'E_disp','Polarizability':'polarizability'})
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']

# Parse XYZ+BO
def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,"Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym,0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80]

def parse_one(xyz_p, bo_p, start_idx=0):
    mols, labels, is_Fe = [], [], []
    with gzip.open(bo_p, "rt") as f: bo_lines = [l.strip() for l in f.readlines()]
    with gzip.open(xyz_p, "rt") as f: xyz_blocks = f.read().strip().split(chr(10))
    bo_blocks, cur = [], []
    for line in bo_lines:
        if line.startswith("CSD_code"):
            if cur: bo_blocks.append(cur); cur = []
        elif line: cur.append(line)
    if cur: bo_blocks.append(cur)
    i, mol_idx = 0, start_idx - 1
    while i < len(xyz_blocks):
        line = xyz_blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts)==1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(xyz_blocks): break; i += 1
            atoms, coords = [], []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts)>=4:
                    try:
                        z = atom_to_z(cparts[0])
                        coords.append([float(x) for x in cparts[1:4]]); atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms)<=1: continue
            bo_mat = None
            if mol_idx < len(bo_blocks):
                bo_mat = np.zeros((n_atoms,n_atoms),dtype=np.float32)
                for bline in bo_blocks[mol_idx]:
                    bparts = bline.split()
                    try: src_idx=int(bparts[0])-1
                    except ValueError: continue
                    j = 2
                    while j+1<len(bparts):
                        try:
                            nbr=int(bparts[j])-1; bo_val=float(bparts[j+1])
                            if 0<=src_idx<n_atoms and 0<=nbr<n_atoms: bo_mat[src_idx,nbr]=bo_val
                            j+=3
                        except (ValueError,IndexError): j+=1
            mols.append({"atoms":np.array(atoms),"coords":np.array(coords,dtype=np.float32),"bo":bo_mat,"csv_idx":mol_idx})
            labels.append(df_y.iloc[mol_idx][targets].astype(float).values)
            is_Fe.append(float(26 in atoms))
    return mols, labels, is_Fe

print('Parsing XYZ+BO files...')
all_m, all_l, all_f = [], [], []
for part in [1,2,3]:
    m,l,f = parse_one(f"data_tmQM/local_vB/tmQM_X{part}.xyz.gz", f"data_tmQM/local_vB/tmQM_X{part}.BO.gz", start_idx=len(all_m))
    all_m.extend(m); all_l.extend(l); all_f.extend(f)
    print(f"  Part {part}: {len(m)} TM")
fe_cmplx = all_m; y_fe = np.array(all_l,dtype=np.float32); is_Fe = all_f
print(f'Total: {len(fe_cmplx)} TM complexes')

# Build hybrid graphs (same as tmQM_Fe_XYZ Cell 4)
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM=[1,2,3,12]; B2I_SM={b:i for i,b in enumerate(BOND_TYPES_SM)}; N_BT_SM=4

def atom_features_sm(atom):
    z=atom.GetAtomicNum(); p,g=ELEMENT_DATA_SM.get(z,(0,0))
    return [z,atom.GetDegree(),atom.GetFormalCharge(),p,g,int(atom.GetIsAromatic()),atom.GetTotalNumHs(),int(z>=21)]

def fast_match(sm_atoms, xyz_atoms):
    N=len(sm_atoms)
    if N==len(xyz_atoms) and all(sm_atoms[i]==xyz_atoms[i] for i in range(N)): return list(range(N))
    sm_to_xyz=[-1]*N; used=set()
    for i,z_sm in enumerate(sm_atoms):
        for j,z_xyz in enumerate(xyz_atoms):
            if z_sm==z_xyz and j not in used: sm_to_xyz[i]=j; used.add(j); break
    return sm_to_xyz

rbf_c=np.linspace(1.0,5.0,16); rbf_sigma=0.2
smiles_by_idx=df_y['SMILES'].tolist()
# === OOM防护: 30k采样 (95k全量会触发cgroup内存限制) ===
N_SAMPLE = 30000
rng = np.random.RandomState(42)
perm = rng.permutation(len(fe_cmplx))[:N_SAMPLE]
fe_cmplx = [fe_cmplx[i] for i in perm]
y_fe = y_fe[perm]  # 同步采样 — 图和目标值对齐!
is_Fe = np.array(is_Fe, dtype=np.float32)[perm].tolist()
print(f'Building graphs (sampled {N_SAMPLE:,} complexes for memory safety)...')
graphs, y_vals, is_Fe_filt, atoms_filt, failed = [], [], [], [], 0
for idx, m in enumerate(tqdm(fe_cmplx, desc='Building')):
    csv_i=m["csv_idx"]; smi=smiles_by_idx[csv_i] if csv_i<len(smiles_by_idx) else ''
    if not isinstance(smi,str) or smi=='': failed+=1; continue
    mol=Chem.MolFromSmiles(smi)
    if mol is None: failed+=1; continue
    N=mol.GetNumAtoms(); sm_atoms=[a.GetAtomicNum() for a in mol.GetAtoms()]
    sm_to_xyz=fast_match(sm_atoms, m['atoms']); coords=m['coords']; bo_mat=m.get('bo')
    xf=[atom_features_sm(a) for a in mol.GetAtoms()]
    if bo_mat is not None:
        for i in range(N):
            j=sm_to_xyz[i]
            if j>=0 and j<bo_mat.shape[0]:
                nb=int((bo_mat[j]>0.05).sum())-1; sb=float(bo_mat[j].sum())-bo_mat[j,j]; mb=float(bo_mat[j].max())
                xf[i].extend([max(0,nb),max(0,sb),max(0,mb)])
            else: xf[i].extend([0,0.0,0.0])
    else:
        for i in range(N): xf[i].extend([0,0.0,0.0])
    x=torch.tensor(xf,dtype=torch.float32)
    src,dst,battr=[],[],[]
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(); bt=int(bond.GetBondType())
        za,zb=sm_atoms[i],sm_atoms[j]
        is_ml=float((za>=21)!=(zb>=21)); is_ring=float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf=[0.0]*N_BT_SM
        if bt in B2I_SM: bf[B2I_SM[bt]]=1.0
        bf.extend([is_ring,is_ml]); d,bo=2.0,0.0
        xi,xj=sm_to_xyz[i],sm_to_xyz[j]
        if xi>=0 and xj>=0 and xi<len(coords) and xj<len(coords): d=float(np.linalg.norm(coords[xi]-coords[xj]))
        if bo_mat is not None and xi>=0 and xj>=0 and xi<bo_mat.shape[0] and xj<bo_mat.shape[1]: bo=float(bo_mat[xi,xj])
        for mu in rbf_c: bf.append(np.exp(-(d-mu)**2/rbf_sigma))
        bf.append(bo); battr.extend([bf,bf])
    ei=torch.tensor([src,dst],dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
    ea=torch.tensor(battr,dtype=torch.float32) if battr else torch.zeros(0,23)
    graphs.append({'x':x,'edge_index':ei,'edge_attr':ea,'num_nodes':N})
    y_vals.append(y_fe[idx]); is_Fe_filt.append(is_Fe[idx]); atoms_filt.append(m['atoms'])

y_all=torch.tensor(y_vals,dtype=torch.float32); is_Fe_t=torch.tensor(is_Fe_filt,dtype=torch.float32)
print(f'Built {len(graphs):,} graphs ({failed} failed)')

mask=~(torch.isnan(y_all).any(dim=1)|torch.isinf(y_all).any(dim=1))
graphs=[g for i,g in enumerate(graphs) if mask[i]]
y_all=y_all[mask]; is_Fe_t=is_Fe_t[mask]; atoms_filt=[a for i,a in enumerate(atoms_filt) if mask[i]]
print(f'After NaN removal: {len(graphs):,}')

y_mean=y_all.mean(dim=0); y_std=y_all.std(dim=0).clamp(min=1e-8); y_norm=(y_all-y_mean)/y_std
idx=np.random.RandomState(42).permutation(len(graphs))
n_tr=int(len(graphs)*0.7); n_va=int(len(graphs)*0.15)
tr_idx=idx[:n_tr]; va_idx=idx[n_tr:n_tr+n_va]; te_idx=idx[n_tr+n_va:]
print(f'Train:{len(tr_idx):,} Val:{len(va_idx):,} Test:{len(te_idx):,}')

def g2dev(g): return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in g.items()}
""", "eq_data")

# Model + train function
add_code("""# ====== Shared Model Components ======
HIDDEN = 384; N_LAYERS = 5; BATCH = 128; EPOCHS = 150; EARLY_STOP = 40

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

def scatter_mean_per_graph(h, batch, n):
    out = torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
    cnt = torch.zeros(n, device=h.device, dtype=h.dtype)
    cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)

def scatter_max_per_graph(h, batch, n):
    out = torch.full((n, h.size(1)), float('-inf'), device=h.device, dtype=h.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce='amax', include_self=False)
    return out

class AblationXYZ(nn.Module):
    def __init__(self, heads=12, use_metal_pool=True, edge_dim=23, node_dim=11):
        super().__init__()
        assert HIDDEN % heads == 0
        self.heads = heads; self.use_metal_pool = use_metal_pool; self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.embed = nn.Linear(node_dim, HIDDEN)
        self.convs = nn.ModuleList([GATConv(HIDDEN, HIDDEN//heads, edge_dim, heads=heads) for _ in range(N_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
        self.drop = nn.Dropout(0.10)
        in_dim = HIDDEN*3 if use_metal_pool else HIDDEN*2
        self.readout = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HIDDEN, HIDDEN//2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HIDDEN//2, 8))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        if self.node_dim < x.size(1):
            x = x[:, :self.node_dim]  # 消融: 只用前node_dim维节点特征
        if ea is not None and ea.size(0) > 0 and self.edge_dim < ea.size(1):
            ea = ea[:, :self.edge_dim]  # 消融: 只用前edge_dim维边特征
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            if self.use_metal_pool:
                mm = (x[:,7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
                return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
            return self.readout(torch.cat([h.mean(0), h.max(0).values])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs); max_h = scatter_max_per_graph(h, batch, n_graphs)
        if self.use_metal_pool:
            mm = (x[:,7] > 0.5)
            mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([mean_h, max_h, mf], dim=1))
        return self.readout(torch.cat([mean_h, max_h], dim=1))

def train_eval(heads, use_metal_pool, use_inv_freq_weight, edge_dim=23, node_dim=11, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    m = AblationXYZ(heads, use_metal_pool, edge_dim=edge_dim, node_dim=node_dim).to(device)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    crit = nn.MSELoss(reduction='none')
    scaler = GradScaler('cuda')
    best_va, best_state, patience = float('inf'), None, 0
    yn = y_norm.to(device)

    # Compute weights
    if use_inv_freq_weight:
        mc = Counter()
        for atoms in atoms_filt:
            for z in set(atoms):
                if z >= 21: mc[z] += 1
        total_c = len(atoms_filt)
        weight_map = {z: min(5.0, total_c/(c+1)) for z, c in mc.items()}
        weights = torch.tensor([max((weight_map.get(z,1.0) for z in set(a) if z>=21), default=1.0) for a in atoms_filt], device=device)
    else:
        weights = torch.ones(len(atoms_filt), device=device)

    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(tr_idx)); tr_loss = 0; nb = 0
        for s in range(0, len(tr_idx), BATCH):
            bidx = tr_idx[perm[s:s+BATCH]].tolist(); bg = [g2dev(graphs[i]) for i in bidx]; by = yn[bidx]
            batched, batch_idx = collate_graphs(bg)
            opt.zero_grad()
            with autocast('cuda'): preds = m(batched, batch_idx); loss = (crit(preds, by).mean(dim=1) * weights[bidx]).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
            tr_loss += loss.item(); nb += 1
        m.eval()
        with torch.no_grad():
            va_preds = []
            for s in range(0, len(va_idx), BATCH):
                bidx = va_idx[s:s+BATCH].tolist(); bg = [g2dev(graphs[i]) for i in bidx]
                batched, batch_idx = collate_graphs(bg)
                with autocast('cuda'): va_preds.append(m(batched, batch_idx).float())
            vp = torch.cat(va_preds, dim=0); va_loss = nn.MSELoss()(vp, yn[va_idx])
        if va_loss < best_va: best_va = va_loss; best_state = {k: v.clone() for k, v in m.state_dict().items()}; patience = 0
        else: patience += 1
        if patience >= EARLY_STOP: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        te_preds = []
        for s in range(0, len(te_idx), BATCH):
            bidx = te_idx[s:s+BATCH].tolist(); bg = [g2dev(graphs[i]) for i in bidx]
            batched, batch_idx = collate_graphs(bg)
            with autocast('cuda'): te_preds.append(m(batched, batch_idx).float())
        tp = torch.cat(te_preds, dim=0).cpu() * y_std + y_mean; tt = y_all[te_idx]
    r2s = [r2_score(tt[:,i].numpy(), tp[:,i].numpy()) for i in range(8)]
    return {'r2': r2s, 'mean_r2': np.mean(r2s), 'params': sum(p.numel() for p in m.parameters())}

results = {}
print('Ready. 3 full-scale ablation experiments.')
print(f'HIDDEN={HIDDEN} N_LAYERS={N_LAYERS} EPOCHS={EPOCHS}')
""", "eq_model")

add_code("""# ===== A: Metal Pooling (on vs off) =====
print('\\n' + '='*60)
print('ABLATION A: Metal-centric pooling (95k full)')
print('='*60)
for mp in [True, False]:
    r = train_eval(heads=8, use_metal_pool=mp, use_inv_freq_weight=False)
    results[f'metal_pool={mp}'] = r
    r2 = r['r2']
    print(f'metal_pool={mp}: Mean R2={r["mean_r2"]:.4f}, dipole={r2[3]:.4f}, metal_q={r2[4]:.4f}, params={r["params"]:,}')""", "eq_a")

add_code("""# ===== B: GAT Heads (8 vs 12) =====
print('\\n' + '='*60)
print('ABLATION B: Attention heads (8 vs 12, 95k full)')
print('='*60)
for h in [8, 12]:
    r = train_eval(heads=h, use_metal_pool=True, use_inv_freq_weight=False)
    results[f'heads={h}'] = r
    r2 = r['r2']
    print(f'heads={h}: Mean R2={r["mean_r2"]:.4f}, dipole={r2[3]:.4f}, metal_q={r2[4]:.4f}, params={r["params"]:,}')""", "eq_b")

add_code("""# ===== C: Inverse Frequency Weighting (on vs off) =====
print('\\n' + '='*60)
print('ABLATION C: Inverse-frequency metal weighting (95k full)')
print('='*60)
for iw in [False, True]:
    r = train_eval(heads=8, use_metal_pool=True, use_inv_freq_weight=iw)
    results[f'inv_freq_weight={iw}'] = r
    r2 = r['r2']
    print(f'inv_freq_weight={iw}: Mean R2={r["mean_r2"]:.4f}, dipole={r2[3]:.4f}, metal_q={r2[4]:.4f}')""", "eq_c")

# ============================================================
# Experiment D: DFT Edge Features (23-dim vs 6-dim SMILES only)
# ============================================================
add_code("""# ===== D: DFT Edge Features (23-dim vs 6-dim SMILES only) =====
print('\\n' + '='*60)
print('ABLATION D: DFT Edge Features — 3D distance + BO vs SMILES only')
print('='*60)
for edim, label in [(23, 'DFT edges (23d)'), (6, 'SMILES only (6d)')]:
    r = train_eval(heads=8, use_metal_pool=True, use_inv_freq_weight=False, edge_dim=edim)
    results[label] = r
    r2 = r['r2']
    print(f'{label}: Mean R2={r["mean_r2"]:.4f}, dipole={r2[3]:.4f}, gap={r2[0]:.4f}, params={r["params"]:,}')
""", "eq_d")

# ============================================================
# Experiment E: RBF Resolution (16 centers vs 8 centers)
# ============================================================
add_code("""# ===== E: RBF Resolution (16 RBF vs 8 RBF) =====
print('\\n' + '='*60)
print('ABLATION E: RBF Resolution — 16 centers vs 8 centers')
print('='*60)
for edim, label in [(23, '16 RBF centers (23d)'), (15, '8 RBF centers (15d)')]:
    r = train_eval(heads=8, use_metal_pool=True, use_inv_freq_weight=False, edge_dim=edim)
    results[label] = r
    r2 = r['r2']
    print(f'{label}: Mean R2={r["mean_r2"]:.4f}, dipole={r2[3]:.4f}, gap={r2[0]:.4f}, params={r["params"]:,}')
""", "eq_e")

# ============================================================
# Experiment F: Mayer BO Full Ablation (node 11→8 + edge 23→22)
# ============================================================
add_code("""# ===== F: Mayer BO Full Ablation (node 11d→8d + edge 23d→22d, keeping 16 RBF) =====
print('\\n' + '='*60)
print('ABLATION F: Mayer BO — Full BO removal (node+edge), keeping 16 RBF distances')
print('='*60)
seeds = [42, 123, 456]
bo_results = {'full': [], 'bofree': []}

for nd, ed, key in [(11, 23, 'full'), (8, 22, 'bofree')]:
    label = 'Full BO (node 11d+edge 23d)' if key == 'full' else 'BO-free (node 8d+edge 22d)'
    for seed in seeds:
        r = train_eval(heads=8, use_metal_pool=True, use_inv_freq_weight=False, edge_dim=ed, node_dim=nd, seed=seed)
        bo_results[key].append(r['r2'])
        print(f'{label} seed={seed}: Mean R2={r["mean_r2"]:.4f}, dipole={r["r2"][3]:.4f}')

bo_full = np.array(bo_results['full'])   # [3, 8]
bo_free = np.array(bo_results['bofree']) # [3, 8]

print(f'\\n{\"=\"*60}')
print('F: BO ABLATION — 3-seed mean +/- std')
print(f'{\"=\"*60}')
print(f'{\"Property\":16s} {\"Full R2\":>14s} {\"BO-free R2\":>14s} {\"Delta\":>10s}')
print('-' * 58)
for i, prop in enumerate(targets):
    fm, fs = bo_full[:, i].mean(), bo_full[:, i].std()
    bm, bs = bo_free[:, i].mean(), bo_free[:, i].std()
    print(f'{prop:16s} {fm:8.4f}+/-{fs:.4f} {bm:8.4f}+/-{bs:.4f} {fm-bm:+10.4f}')

full_mean = bo_full.mean(axis=1).mean(); full_std = bo_full.mean(axis=1).std()
free_mean = bo_free.mean(axis=1).mean(); free_std = bo_free.mean(axis=1).std()
print(f'\\nFull BO Mean R2:    {full_mean:.4f} +/- {full_std:.4f}')
print(f'BO-free Mean R2:     {free_mean:.4f} +/- {free_std:.4f}')
print(f'Delta Mean R2:       {full_mean-free_mean:+.4f}')
results['BO ablation full (3-seed)'] = {'r2': bo_full.mean(axis=0).tolist(), 'mean_r2': float(full_mean), 'params': r['params']}
results['BO ablation free (3-seed)'] = {'r2': bo_free.mean(axis=0).tolist(), 'mean_r2': float(free_mean), 'params': r['params']}
""", "eq_f")

add_code("""# ===== SUMMARY =====
print('\\n' + '='*80)
print('FULL-SCALE ABLATION RESULTS (95k complexes)')
print('='*80)
print(f'{"Experiment":30s} {"Mean R2":>8s} {"gap":>7s} {"homo":>7s} {"lumo":>7s} {"dipole":>7s} {"metal_q":>8s} {"params":>8s}')
print('-'*80)
for name, r in results.items():
    r2 = r['r2']
    print(f'{name:30s} {r["mean_r2"]:8.4f} {r2[0]:7.4f} {r2[1]:7.4f} {r2[2]:7.4f} {r2[3]:7.4f} {r2[4]:8.4f} {r["params"]:>8,}')
with open('ablation_full_results.json', 'w') as f:
    json.dump({k: {'mean_r2': v['mean_r2'], 'r2': [float(x) for x in v['r2']], 'params': v['params']} for k, v in results.items()}, f, indent=2)
print('\\nSaved to ablation_full_results.json')""", "eq_summary")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('ablation_full.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
with open('ablation_full.ipynb', encoding='utf-8') as f:
    json.load(f)
print(f'Built {len(cells)} cells, JSON valid')
