"""SMILES GAT on 5 DFT-validated catalysts — quick inference"""
import numpy as np, torch, torch.nn as nn
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

device = torch.device('cpu')

# ===== SMILES GAT model (same as tmQM_Fe.ipynb) =====
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10

ELEMENT_DATA = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES = [1,2,3,12]; B2I = {b:i for i,b in enumerate(BOND_TYPES)}; N_BT = len(BOND_TYPES)

def atom_features(atom):
    z = atom.GetAtomicNum(); p,g = ELEMENT_DATA.get(z,(0,0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g, int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    N = x.size(0); src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); bt = int(bond.GetBondType())
        za, zb = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za>=21)!=(zb>=21)); is_ring = float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf = [0.0]*N_BT
        if bt in B2I: bf[B2I[bt]] = 1.0
        bf.extend([is_ring, is_ml]); battr.extend([bf,bf])
    ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, N_BT+2)
    return {'x':x, 'edge_index':ei, 'edge_attr':ea, 'num_nodes':N}

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

class SMILES_GAT(nn.Module):
    def __init__(self, node_dim=8, edge_dim=6, hidden=HIDDEN, n_layers=N_LAYERS, output_dim=8, dropout=DROPOUT, heads=GAT_HEADS):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            mm = (x[:,7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n)
        max_h = scatter_max_per_graph(h, batch, n)
        mm = (x[:,7] > 0.5); mf = scatter_mean_per_graph(h[mm], batch[mm], n) if mm.any() else torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

# ===== Load model =====
model = SMILES_GAT().to(device)
ckpt = 'tmqm_gat_best.pt'
state = torch.load(ckpt, map_location=device, weights_only=False)
model.load_state_dict(state, strict=False)
model.eval()
print(f'Loaded {ckpt}')

# y_mean/y_std from full tmQM
y_mean = torch.tensor([0.108505, -0.197591, -0.089089, 5.799378, 0.162882, -2987.275391, -0.142229, 397.123474])
y_std  = torch.tensor([0.033575,  0.053824,  0.054516, 3.927464, 0.782902,  1634.063721,  0.070114, 152.458313])

# ===== 5 catalysts =====
catalysts = {
    'Co-cat1': 'CC1=O[Co]2(OC(C)=O2)O1',
    'Pd-cat1': 'CC1(C)C2=C(OC3=C4C=CC=C13)C([P]([Pd][P](C5=CC=CC=C5)4C6=CC=CC=C6)(C7=CC=CC=C7)C8=CC=CC=C8)=CC=C2',
    'Ni-cat3': 'I[Ni]1[N]2=C3C4=[N]1C=CC(C5=CC=C(C(F)(F)F)C=C5)=C4C=CC3=C(C6=CC=C(C(F)(F)F)C=C6)C=C2',
    'Fe-cat1': 'O=[N+]1C(C)(C)CCCC1(C)C.Cl[Fe](Cl)Cl',
    'Zr-cat3': 'I[Zr]([CH-]1C=CC=C1)([CH-]1C=CC=C1)I',
}

print(f'\n{"Catalyst":10s} {"Gap(eV)":>8s} {"Dipole(D)":>10s} {"HOMO(Ha)":>10s} {"LUMO(Ha)":>10s}')
print('-' * 55)

for name, smi in catalysts.items():
    g = smiles_to_graph(smi)
    g_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.items()}
    with torch.no_grad():
        pred = model(g_gpu).cpu() * y_std + y_mean
    gap = pred[0,0].item()
    gap_ev = gap * 27.2114
    homo = pred[0,1].item()
    lumo = pred[0,2].item()
    dipole = pred[0,3].item()
    print(f'{name:10s} {gap_ev:8.2f} {dipole:10.2f} {homo:10.4f} {lumo:10.4f}')
