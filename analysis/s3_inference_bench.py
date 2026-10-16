# -*- coding: utf-8 -*-
"""S3: 推理延迟/吞吐量基准（当前环境实测；GPU 数字留待训练机补测）
不涉及训练。输出 s3_inference_bench.json。"""
import time
import json
import platform
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(16)
device = torch.device('cpu')
print(f'Device: {device}; threads: {torch.get_num_threads()}')

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)


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
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0:
            attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, N)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(N, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out)


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


class TMetalGNN(nn.Module):
    def __init__(self, node_dim=8, edge_dim=6, hidden=384, n_layers=5, output_dim=8, dropout=0.10, heads=12):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden // heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            mm = (x[:, 7] > 0.5)
            mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        mm = (x[:, 7] > 0.5)
        mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))


ELEMENT_DATA = {1: (1, 1), 5: (2, 13), 6: (2, 14), 7: (2, 15), 8: (2, 16), 9: (2, 17), 14: (3, 14),
                15: (3, 15), 16: (3, 16), 17: (3, 17), 20: (4, 2), 21: (4, 3), 22: (4, 4), 23: (4, 5),
                24: (4, 6), 25: (4, 7), 26: (4, 8), 27: (4, 9), 28: (4, 10), 29: (4, 11), 30: (4, 12),
                35: (4, 17), 42: (5, 6), 44: (5, 8), 45: (5, 9), 46: (5, 10), 47: (5, 11), 48: (5, 12),
                53: (5, 17), 72: (6, 4), 73: (6, 5), 74: (6, 6), 75: (6, 7), 76: (6, 8), 77: (6, 9),
                78: (6, 10), 79: (6, 11), 80: (6, 12)}
BOND_TYPES = [1, 2, 3, 12]; B2I = {b: i for i, b in enumerate(BOND_TYPES)}; N_BT = 4
RBF_CENTERS = np.linspace(1.0, 5.0, 16); RBF_SIGMA = 0.2


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
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 6)
    return {'x': x, 'edge_index': ei, 'edge_attr': ea}


def smiles_to_3d_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200) != 0: return None
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        return None
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    N = mol.GetNumAtoms()
    xf = [atom_features(a) for a in mol.GetAtoms()]
    for i in range(N): xf[i].extend([0, 0.0, 0.0])
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, battr = [], [], []
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] for i in range(N)])
    sm_z = [xf[i][0] for i in range(N)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = sm_z[i], sm_z[j]
        is_ml = float((za >= 21) != (zb >= 21)); is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * 4
        if bt in B2I: bf[B2I[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        d = float(np.linalg.norm(coords[i] - coords[j]))
        for mu in RBF_CENTERS: bf.append(np.exp(-(d - mu) ** 2 / RBF_SIGMA))
        bf.append(0.0)
        battr.extend([bf, bf])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 23)
    return {'x': x, 'edge_index': ei, 'edge_attr': ea}


def collate_cpu(gs):
    x_list = [g['x'] for g in gs]; ei_list = [g['edge_index'] for g in gs]; ea_list = [g['edge_attr'] for g in gs]
    batch_list = []; offset = 0
    for k, x in enumerate(x_list):
        n = x.size(0)
        batch_list.append(torch.full((n,), k, dtype=torch.long))
        ei_list[k] = ei_list[k] + offset
        offset += n
    return {'x': torch.cat(x_list, dim=0), 'edge_index': torch.cat(ei_list, dim=1),
            'edge_attr': torch.cat(ea_list, dim=0)}, torch.cat(batch_list, dim=0)


# ---------- 模型加载 ----------
sm_state = torch.load('tmqm_gat_best.pt', map_location='cpu', weights_only=False)
sm_state = {k.replace('_orig_mod.', ''): v for k, v in sm_state.items()}
model_sm = TMetalGNN(8, 6); model_sm.load_state_dict(sm_state); model_sm.eval()
n_sm = sum(p.numel() for p in model_sm.parameters())

st3 = torch.load('xyz_best_weighted.pt', map_location='cpu', weights_only=False)
st3 = {k.replace('_orig_mod.', ''): v for k, v in st3.items()}
model_3d = TMetalGNN(11, 23); model_3d.load_state_dict(st3); model_3d.eval()
n_3d = sum(p.numel() for p in model_3d.parameters())
print(f'SMILES GAT params: {n_sm:,}; 3D GAT params: {n_3d:,}')

# ---------- 代表分子 ----------
TEST_SM = 'Br[Ni]1([N]2=C3C4=[N]1C=CC(C5=CC=C(C(F)(F)F)C=C5)=C4C=CC3=C(C6=CC=C(C(F)(F)F)C=C6)C=C2)Br'  # Ni-cat1
g_sm = smiles_to_graph(TEST_SM)
g_3d = smiles_to_3d_graph(TEST_SM)
print(f'Ni-cat1 atoms: SMILES graph {g_sm["x"].size(0)}, 3D graph {g_3d["x"].size(0)}')

out = {'env': {'torch': torch.__version__, 'rdkit': rdkit.__version__,
               'python': platform.python_version(), 'os': platform.platform(),
               'cpu': platform.processor() or platform.machine(), 'threads': torch.get_num_threads()},
       'params': {'smiles_gat': n_sm, '3d_gat': n_3d},
       'molecule': {'name': 'Ni-cat1', 'atoms_sm': g_sm['x'].size(0)}}


def bench(fn, warmup=30, iters=200):
    with torch.no_grad():
        for _ in range(warmup): fn()
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        dt = (time.perf_counter() - t0) / iters
    return dt


with torch.no_grad():
    t_sm1 = bench(lambda: model_sm(g_sm))
    out['latency_ms'] = {'smiles_gat_batch1': round(t_sm1 * 1000, 3)}
    print(f'SMILES GAT batch=1: {t_sm1*1000:.3f} ms')
    t_3d1 = bench(lambda: model_3d(g_3d))
    out['latency_ms']['3d_gat_batch1'] = round(t_3d1 * 1000, 3)
    print(f'3D GAT batch=1: {t_3d1*1000:.3f} ms')

    # batch=128 吞吐（SMILES GAT，用 Ni-cat1 图复制 128 份）
    gs = [smiles_to_graph(TEST_SM) for _ in range(128)]
    b, bi = collate_cpu(gs)
    t0 = time.perf_counter()
    for _ in range(20): model_sm(b, bi)
    dt = (time.perf_counter() - t0) / 20
    out['throughput'] = {'smiles_gat_batch128_ms': round(dt * 1000, 2),
                         'complexes_per_sec_batch128': round(128 / dt, 1)}
    print(f'SMILES GAT batch=128: {dt*1000:.2f} ms ({128/dt:.1f} complexes/s)')
    # batch=128 吞吐（3D GAT，复用同一图 128 份，避免重复 ETKDG）
    gs3 = [g_3d] * 128
    b3, bi3 = collate_cpu(gs3)
    t0 = time.perf_counter()
    for _ in range(20): model_3d(b3, bi3)
    dt3 = (time.perf_counter() - t0) / 20
    out['throughput']['3d_gat_batch128_ms'] = round(dt3 * 1000, 2)
    out['throughput']['3d_complexes_per_sec_batch128'] = round(128 / dt3, 1)
    print(f'3D GAT batch=128: {dt3*1000:.2f} ms ({128/dt3:.1f} complexes/s)')

# ETKDG+MMFF 生成延迟（Ni-cat1）
def etkdg_once():
    mol = Chem.MolFromSmiles(TEST_SM)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

t0 = time.perf_counter()
for _ in range(5): etkdg_once()
t_et = (time.perf_counter() - t0) / 5
out['etkdg_mmff_ms'] = round(t_et * 1000, 1)
print(f'ETKDG+MMFF (Ni-cat1, {g_3d["x"].size(0)} atoms): {t_et*1000:.1f} ms')

json.dump(out, open('s3_inference_bench.json', 'w'), indent=1)
print('\nsaved s3_inference_bench.json')
