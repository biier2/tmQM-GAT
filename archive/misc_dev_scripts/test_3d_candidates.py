"""Test which candidate catalysts can use 3D GAT (ETKDG success + positive gap)"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
import warnings; warnings.filterwarnings('ignore')

device = torch.device('cpu')
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10
ELEMENT_DATA = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES = [1,2,3,12]; B2I = {b:i for i,b in enumerate(BOND_TYPES)}; N_BT = len(BOND_TYPES)
RBF_CENTERS = np.linspace(1.0, 5.0, 16); RBF_SIGMA = 0.2

def atom_features(atom):
    z = atom.GetAtomicNum(); p,g = ELEMENT_DATA.get(z,(0,0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g, int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

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
        attn = F.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, N)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(N, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out)

class ThreeD_GAT(nn.Module):
    def __init__(self):
        super().__init__(); self.heads = GAT_HEADS
        self.embed = nn.Linear(11, HIDDEN)
        self.convs = nn.ModuleList([GATConv(HIDDEN, HIDDEN//GAT_HEADS, 23, heads=GAT_HEADS) for _ in range(N_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
        self.drop = nn.Dropout(DROPOUT)
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN*3, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN//2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN//2, 8))
    def forward(self, g):
        x, ei, ea = g['x'], g['edge_index'], g['edge_attr']
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        mm = (x[:,7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(HIDDEN)
        return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)

model_3d = ThreeD_GAT().to(device)
state = torch.load('xyz_best_weighted.pt', map_location='cpu', weights_only=False)
clean = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
model_3d.load_state_dict(clean); model_3d.eval()

y_mean = torch.tensor([0.108505, -0.197591, -0.089089, 5.799378, 0.162882, -2987.275391, -0.142229, 397.123474])
y_std  = torch.tensor([0.033575,  0.053824,  0.054516, 3.927464, 0.782902,  1634.063721,  0.070114, 152.458313])

def threed_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200) != 0: return None
        AllChem.MMFFOptimizeMolecule(mol)
    except: return None
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer(); N = mol.GetNumAtoms()
    xf = [atom_features(a) for a in mol.GetAtoms()]
    for i in range(N): xf[i].extend([0, 0.0, 0.0])
    x = torch.tensor(xf, dtype=torch.float32)
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(N)])
    sm_z = [xf[i][0] for i in range(N)]
    src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); bt = int(bond.GetBondType())
        za, zb = sm_z[i], sm_z[j]
        is_ml = float((za>=21)!=(zb>=21)); is_ring = float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf = [0.0]*N_BT
        if bt in B2I: bf[B2I[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        d = float(np.linalg.norm(coords[i]-coords[j]))
        for mu in RBF_CENTERS: bf.append(np.exp(-(d-mu)**2/RBF_SIGMA))
        bf.append(0.0); battr.extend([bf,bf])
    ei = torch.tensor([src,dst], dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32)
    return {'x': x, 'edge_index': ei, 'edge_attr': ea}

# Candidate catalysts — Pd/Pt/Rh phosphine family (ETKDG-friendly)
PPh3 = 'P(c1ccccc1)(c2ccccc2)c3ccccc3'
PM3 = 'P(C)(C)C'
dppe = '[P](c1ccccc1)(c2ccccc2)CC[P](c3ccccc3)(c4ccccc4)'  # bidentate

candidates = {
    # Pd phosphine family
    'Pd-cat1 (existing)': 'CC1(C)C2=C(OC3=C4C=CC=C13)C([P]([Pd][P](C5=CC=CC=C5)4C6=CC=CC=C6)(C7=CC=CC=C7)C8=CC=CC=C8)=CC=C2',
    'cis-PdCl2(PPh3)2': f'Cl[Pd@SP1](Cl)({PPh3}){PPh3}',
    'trans-PdCl2(PPh3)2': f'Cl[Pd@SP2](Cl)({PPh3}){PPh3}',
    'Pd(dppe)Cl2': f'Cl[Pd](Cl)({dppe})',
    'Pd(acac)2-like': 'CC(=O)C(C(C)=O)[Pd]C(C(C)=O)C(C)=O',
    # Pt family
    'cis-PtCl2(PPh3)2': f'Cl[Pt@SP1](Cl)({PPh3}){PPh3}',
    'trans-PtCl2(PPh3)2': f'Cl[Pt@SP2](Cl)({PPh3}){PPh3}',
    'Pt(dppe)Cl2': f'Cl[Pt](Cl)({dppe})',
    # Rh family
    'RhCl(PPh3)3': f'Cl[Rh]({PPh3})({PPh3}){PPh3}',
    # Cu
    'CuCl(PPh3)3': f'Cl[Cu]({PPh3})({PPh3}){PPh3}',
    # Ru bipy
    'RuCl2(bpy)2': 'Cl[Ru]1(Cl)([n]2cccc3ccc[n]1c23)[n]4cccc5ccc[n]1c45',
}

print(f'{"Catalyst":28s} {"ETKDG":>6s} {"gap(Ha)":>9s} {"gap(eV)":>8s} {"3D GAT OK":>10s}')
print('-'*70)
ok_count = 0
for name, smi in candidates.items():
    g = threed_graph(smi)
    if g is None:
        print(f'{name:28s} {"FAIL":>6s}')
        continue
    with torch.no_grad():
        pred = (model_3d(g) * y_std + y_mean).numpy()[0]
    gap_ha = pred[0]; gap_ev = gap_ha * 27.2114
    ok = gap_ha > 0
    if ok: ok_count += 1
    print(f'{name:28s} {"OK":>6s} {gap_ha:9.4f} {gap_ev:8.2f} {"YES" if ok else "NO":>10s}')

print(f'\n{ok_count}/{len(candidates)} candidates can use 3D GAT')
