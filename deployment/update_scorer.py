"""Rewrite scorer_3d_only.ipynb — single 3D GAT model"""
import json

with open(r'd:\ml\chem_ml6\scorer_3d_only.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Update intro
for cell in nb['cells']:
    if cell.get('id') == '407e2506':
        cell['source'] = [l + '\n' for l in [
            '# 3D GAT 分子打分器',
            '',
            '## 单一模型：3D-enhanced GAT 预测全部 8 项性质',
            '',
            '```',
            '候选 SMILES -> RDKit 建图 (SMILES bonds)',
            '            -> ETKDG+MMFF 生成 3D 坐标',
            '            -> 计算 RBF 距离编码 (16 centers, 1.0-5.0 A)',
            '            -> 3D GAT 预测全部 8 项性质',
            '```',
            '',
            '- 节点: 11 维 (SMILES 8d + BO 统计 3d, BO=0 at inference)',
            '- 边: 23 维 (SMILES bonds 6d + RBF 16d + BO 1d, BO=0 at inference)',
            '- 模型: 12-head GAT x 384d x 5 layers + metal-centric pooling'
        ]]
        break

# Cell 2: Replace SMILES GNN with placeholder
for cell in nb['cells']:
    if cell.get('id') == '34fb7ed2':
        cell['source'] = [l + '\n' for l in [
            '# ============================================================',
            '# Step 2: 3D GAT 单一模型 (SMILES GAT 已移除, 3D GAT 统一预测)',
            '# ============================================================',
            '# 所有 8 项性质均由 3D GAT 预测',
            '# 坐标由 ETKDG+MMFF 自动生成, 无需外部文件',
            '# Mayer BO 在推理时设为 0'
        ]]
        break

# Cell 3: Replace XYZ model with full 3D GAT
g3d_source = [
    '# ============================================================',
    '# Step 3: 3D GAT 模型 + SMILES->3D图构建 (ETKDG+MMFF, BO=0)',
    '# ============================================================',
    '',
    'import os',
    "CKPT_3D = '../chem_ml5/xyz_best_weighted.pt'",
    "if not os.path.exists(CKPT_3D):",
    "    CKPT_3D = '../chem_ml5/xyz_best.pt'  # fallback",
    '',
    'import torch.nn.functional as F',
    '',
    'HIDDEN=384; N_LAYERS=5; GAT_HEADS=12; DROPOUT=0.10',
    '',
    'def segment_softmax(scores, dst, N):',
    "    scores_f32 = scores.float()",
    "    max_v = torch.full((N, scores_f32.size(1)), float('-inf'), device=scores_f32.device, dtype=torch.float32)",
    "    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), scores_f32, reduce='amax', include_self=False)",
    "    exp_s = torch.exp(scores_f32 - max_v[dst])",
    "    sum_v = torch.zeros(N, scores_f32.size(1), device=scores_f32.device, dtype=torch.float32)",
    "    sum_v.scatter_add_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), exp_s)",
    "    return (exp_s / (sum_v[dst] + 1e-8)).to(scores.dtype)",
    '',
    'class GATConv(nn.Module):',
    '    def __init__(self, in_dim, out_dim_per_head, edge_dim=0, heads=4):',
    '        super().__init__(); self.heads, self.out_dim = heads, out_dim_per_head',
    '        total_dim = out_dim_per_head * heads',
    '        self.node_lin = nn.Linear(in_dim, total_dim, bias=False)',
    '        self.attn_src = nn.Linear(total_dim, heads, bias=False)',
    '        self.attn_dst = nn.Linear(total_dim, heads, bias=False)',
    '        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim>0 else None',
    '        self.act = nn.ReLU()',
    '    def forward(self, x, ei, ea=None):',
    '        src, dst = ei[0], ei[1]; N = x.size(0); h = self.node_lin(x)',
    '        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])',
    '        if self.edge_attn is not None and ea is not None and ea.size(0)>0: attn = attn + self.edge_attn(ea)',
    '        attn = F.leaky_relu(attn, 0.2)',
    '        alpha = segment_softmax(attn, dst, N)',
    '        h_heads = h.view(-1, self.heads, self.out_dim)',
    '        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)',
    '        out = torch.zeros(N, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)',
    '        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)',
    '        return self.act(out)',
    '',
    'class ThreeD_GAT(nn.Module):',
    '    def __init__(self, node_dim=11, edge_dim=23, hidden=HIDDEN, n_layers=N_LAYERS,',
    '                 output_dim=8, dropout=DROPOUT, heads=GAT_HEADS):',
    '        super().__init__(); self.heads = heads',
    '        self.embed = nn.Linear(node_dim, hidden)',
    '        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, edge_dim, heads=heads) for _ in range(n_layers)])',
    '        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])',
    '        self.drop = nn.Dropout(dropout)',
    '        self.readout = nn.Sequential(',
    '            nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(dropout),',
    '            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),',
    '            nn.Linear(hidden//2, output_dim))',
    '    def forward(self, g):',
    "        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)",
    '        h = self.embed(x)',
    '        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)',
    '        mm = (x[:,7] > 0.5)',
    '        mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)',
    '        return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)',
    '',
    'model_3d = ThreeD_GAT().to(device)',
    'if os.path.exists(CKPT_3D):',
    '    state = torch.load(CKPT_3D, map_location=device, weights_only=False)',
    "    # Strip torch.compile prefix if present",
    "    clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}",
    "    model_3d.load_state_dict(clean_state)",
    "    print(f'3D GAT loaded from {CKPT_3D}')",
    'else:',
    "    print(f'WARNING: checkpoint not found ({CKPT_3D})')",
    'model_3d.eval()',
    "print(f'Params: {sum(p.numel() for p in model_3d.parameters()):,}')",
    '',
    '# ===== SMILES -> 3D 图构建 (ETKDG + MMFF, RBF=16, BO=0) =====',
    'import numpy as np',
    'RBF_CENTERS = np.linspace(1.0, 5.0, 16)',
    'RBF_SIGMA = 0.2',
    '',
    'def smiles_to_3d_graph(smiles):',
    '    """SMILES -> full 3D GAT graph (RBF distances, BO=0)"""',
    '    mol = Chem.MolFromSmiles(smiles)',
    '    if mol is None: return None',
    '    mol = Chem.AddHs(mol)',
    '    try:',
    '        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200) != 0:',
    '            return None',
    '        AllChem.MMFFOptimizeMolecule(mol)',
    '    except:',
    '        return None',
    '    mol = Chem.RemoveHs(mol)',
    '    conf = mol.GetConformer()',
    '    N = mol.GetNumAtoms()',
    '',
    '    # Node: SMILES 8d + BO stats 3d (all zero)',
    '    xf = [atom_features(a) for a in mol.GetAtoms()]',
    '    for i in range(N): xf[i].extend([0, 0.0, 0.0])',
    '    x = torch.tensor(xf, dtype=torch.float32)',
    '',
    '    # Edge: SMILES bonds 6d + RBF 16d + BO 1d',
    '    src, dst, battr = [], [], []',
    '    coords = np.array([[conf.GetAtomPosition(i).x,',
    '                        conf.GetAtomPosition(i).y,',
    '                        conf.GetAtomPosition(i).z] for i in range(N)])',
    '    sm_z = [xf[i][0] for i in range(N)]',
    '',
    '    for bond in mol.GetBonds():',
    '        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()',
    '        bt = int(bond.GetBondType())',
    '        za, zb = sm_z[i], sm_z[j]',
    '        is_ml = float((za>=21)!=(zb>=21))',
    '        is_ring = float(bond.IsInRing())',
    '        src.extend([i,j]); dst.extend([j,i])',
    '        bf = [0.0]*4',
    '        if bt in B2I: bf[B2I[bt]] = 1.0',
    '        bf.extend([is_ring, is_ml])',
    '        d = float(np.linalg.norm(coords[i]-coords[j]))',
    '        for mu in RBF_CENTERS:',
    '            bf.append(np.exp(-(d-mu)**2/RBF_SIGMA))',
    '        bf.append(0.0)  # Mayer BO = 0',
    '        battr.extend([bf,bf])',
    '',
    '    ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)',
    '    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0,23)',
    "    return {'x': x, 'edge_index': ei, 'edge_attr': ea}",
    '',
    "print('3D GAT scorer ready (SMILES -> ETKDG+MMFF -> RBF -> 3D GAT -> 8 props)')"
]

for cell in nb['cells']:
    if cell.get('id') == '04b0d241':
        cell['source'] = [l + '\n' for l in g3d_source]
        break

# Cell 4: Simplify scoring
scoring_source = [
    '# ============================================================',
    '# Step 4: 填入候选 + 3D GAT 打分',
    '# ============================================================',
    '',
    '# ↓↓↓ 在这里填入你的候选 SMILES ↓↓↓',
    'candidates = {',
    "    'Fe_dichloro_nitrato_diol_a': 'Cl[Fe]Cl.O[N+](=O)[O-].[Fe].OCCCO',",
    "    'Fe_dichloro_nitrato_diol_b': 'Cl[Fe](Cl)(O[N+](=O)[O-])(OCCCO)[Fe]',",
    '}',
    '',
    "targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']",
    "print(f'Scoring {len(candidates)} candidates with 3D GAT...\\n')",
    '',
    'results = []',
    'for name, smi in candidates.items():',
    '    g = smiles_to_3d_graph(smi)',
    '    if g is None:',
    "        print(f'  SKIP {name}: conformer generation failed')",
    '        continue',
    "    g = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.items()}",
    '    with torch.no_grad():',
    '        pred = model_3d(g).cpu().numpy().flatten()',
    '',
    '    results.append({',
    "        'Name': name,",
    "        'gap_Ha':    pred[0],",
    "        'homo_Ha':   pred[1],",
    "        'lumo_Ha':   pred[2],",
    "        'dipole_D':  pred[3],",
    "        'metal_q_e': pred[4],",
    "        'E_elec_Ha': pred[5],",
    "        'E_disp_Ha': pred[6],",
    "        'polariz_a03': pred[7],",
    '    })',
    "    print(f'  {name}: gap={pred[0]:.4f} Ha, dipole={pred[3]:.2f} D')",
    '',
    "df = pd.DataFrame(results).sort_values('E_elec_Ha')",
    "print(f'\\n=== Results ({len(results)} candidates scored) ===')",
    'print(df.to_string(index=False))',
    '',
    'if len(results) > 0:',
    '    best = df.iloc[0]',
    "    print(f'\\nBest: {best[\"Name\"]}')",
    "    print(f'  gap    = {best[\"gap_Ha\"]:.4f} Ha = {best[\"gap_Ha\"]*27.2114:.2f} eV')",
    "    print(f'  dipole = {best[\"dipole_D\"]:.2f} D')",
    "    print(f'  E_elec = {best[\"E_elec_Ha\"]:.1f} Ha')"
]

for cell in nb['cells']:
    if cell.get('id') == '57648326':
        cell['source'] = [l + '\n' for l in scoring_source]
        break

# Cell 5: Update plot labels
for cell in nb['cells']:
    if cell.get('id') == '5eecddd3':
        src = ''.join(cell['source'])
        src = src.replace("('Candidate Scoring (SMILES GAT + XYZ dipole)'",
                          "('Candidate Scoring (3D GAT, single model)'")
        src = src.replace("'dipole_D(xyz)'", "'dipole_D'")
        src = src.replace("'Dipole from XYZ+BO GNN (Debye)'", "'Dipole from 3D GAT (Debye)'")
        cell['source'] = [l + '\n' for l in src.split('\n')]
        break

# Cell 6: Update explanation
for cell in nb['cells']:
    if cell.get('id') == 'ca4a2389':
        cell['source'] = [l + '\n' for l in [
            '## 输出说明',
            '',
            '| 列 | 来源模型 | 说明 |',
            '|----|---------|------|',
            '| 全部 8 项性质 | **3D GAT** | 单一模型预测所有性质 |',
            '| E_elec_Ha | 3D GAT | 按此项排序, 越低越稳定 |',
            '| dipole_D | 3D GAT | 基于 ETKDG+MMFF 生成坐标的 RBF 距离编码 |',
            '| gap_Ha / homo_Ha / lumo_Ha | 3D GAT | 前线轨道能量 |',
            '',
            '> 输入仅需 SMILES 字符串。3D 坐标由 ETKDG 构象搜索 + MMFF 力场优化自动生成。',
            '> Mayer bond orders 在推理时设为 0。',
            '> 使用 12-head GAT x 384d x 5 layers checkpoint。'
        ]]
        break

with open(r'd:\ml\chem_ml6\scorer_3d_only.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved scorer_3d_only.ipynb')
