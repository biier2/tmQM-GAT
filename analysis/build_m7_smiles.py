# -*- coding: utf-8 -*-
"""Generate the SMILES version from m7_carbon_donors.py:
- 8d nodes / 6d edges (RBF/BO removed), checkpoint tmqm_gat_best.pt
- additionally append B3: attention vs perturbation top-5 overlap + paired z test (5/N baseline)
- output m7_carbon_donors_smiles.json (including the overlap data)"""
import re

src = open('m7_carbon_donors.py', encoding='utf-8').read()

# ---- 1. docstring ----
src = src.replace(
    '"""M7: recompute with an extended donor set covering carbon donors (NHC carbene C / CO carbonyl C / other M-C)\nProtocol (reproducing xai_rigorous.ipynb):\n  - XAI subset: RandomState(42), graphs built over the full permutation up to 20,000 (11d node / 23d edge, RBF placeholder 2.0A, MBO 0)\n  - Model: xyz_best.pt = XYZ_GNN_XAI (11->384, 5xGATConv 12heads 32d, edge 23d, readout 3x384->8)',
    '"""M7-SMILES: recompute with an extended donor set covering carbon donors (SMILES deployment model version)\nProtocol (reproducing xai_smiles.ipynb):\n  - XAI subset: RandomState(42), graphs built over the full permutation up to 20,000 (8d node / 6d edge)\n  - Model: tmqm_gat_best.pt = deployed SMILES GAT (8->384, 5xGATConv 12heads 32d, edge 6d, readout 3x384->8)')

# ---- 2. node features 11d -> 8d ----
src = src.replace("xf.append([z, a.GetDegree(), a.GetFormalCharge(), p, g, int(a.GetIsAromatic()), a.GetTotalNumHs(), int(z >= 21), 0, 0.0, 0.0])",
                  "xf.append([z, a.GetDegree(), a.GetFormalCharge(), p, g, int(a.GetIsAromatic()), a.GetTotalNumHs(), int(z >= 21)])  # SMILES 8d")

# ---- 3. edge features 23d -> 6d (remove the RBF loop and the MBO placeholder) ----
src = src.replace("""        for mu in rbf_c: bf.append(np.exp(-(2.0 - mu) ** 2 / 0.2))
        bf.append(0.0); battr.extend([bf, bf])""",
                  """        battr.extend([bf, bf])  # SMILES 6d (no RBF/MBO)""")
src = src.replace("ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 23)",
                  "ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 6)")

# ---- 4. model dimensions ----
src = src.replace("self.embed = nn.Linear(11, HIDDEN)", "self.embed = nn.Linear(8, HIDDEN)  # SMILES 8d")
src = src.replace("GATConvXAI(HIDDEN, HIDDEN // GAT_HEADS, 23, heads=GAT_HEADS) for _ in range(N_LAYERS)])",
                  "GATConvXAI(HIDDEN, HIDDEN // GAT_HEADS, 6, heads=GAT_HEADS) for _ in range(N_LAYERS)])  # SMILES 6d")

# ---- 5. checkpoint ----
src = src.replace("state = torch.load('xyz_best.pt', map_location='cpu')", "state = torch.load('tmqm_gat_best.pt', map_location='cpu')")
src = src.replace("print(f'xyz_best.pt loaded; missing={missing[:3]} unexpected={unexpected[:3]}')",
                  "print(f'tmqm_gat_best.pt loaded; missing={missing[:3]} unexpected={unexpected[:3]}')")

# ---- 6. output json name ----
src = src.replace("with open('m7_carbon_donors.json', 'w', encoding='utf-8') as f:",
                  "with open('m7_carbon_donors_smiles.json', 'w', encoding='utf-8') as f:")
src = src.replace("print('\\nSaved -> m7_carbon_donors.json')", "print('\\nSaved -> m7_carbon_donors_smiles.json')")

# ---- 7. append B3: overlap + z test (inserted before the json save) ----
b3 = '''
# ===== 5. attention vs perturbation top-5 consistency + paired z test =====
print('\\n=== B3: attention vs perturbation top-5 overlap + z test ===')
overlap_rates, chance_rates = [], []
for ii in tqdm(range(len(perturb_idx)), desc='consistency'):
    i = perturb_idx[ii]
    g = xai_graphs[i]; N = g['num_nodes']
    with torch.no_grad():
        _, attn_layers = model(g, return_attn=True)
    alpha = attn_layers[-1].mean(dim=1)
    dst = g['edge_index'][1]
    node_attn = torch.zeros(N, device=device)
    node_attn.scatter_add_(0, dst, alpha)
    node_attn = node_attn / (node_attn.sum() + 1e-8)
    attn_top5 = set(node_attn.argsort(descending=True)[:5].numpy())
    x_orig = g['x']
    with torch.no_grad():
        base_pred = model(g).cpu().numpy()[0]
    total_impact = np.zeros(N)
    for atom_idx in range(N):
        x_masked = x_orig.clone()
        x_masked[atom_idx] = 0.0
        with torch.no_grad():
            mp = model({'x': x_masked, 'edge_index': g['edge_index'], 'edge_attr': g['edge_attr']}).cpu().numpy()[0]
        total_impact[atom_idx] = np.abs(base_pred - mp).sum()
    pert_top5 = set(np.argsort(-total_impact)[:5])
    overlap_rates.append(len(attn_top5 & pert_top5) / 5)
    chance_rates.append(5.0 / N)
overlap_rates = np.array(overlap_rates); chance_rates = np.array(chance_rates)
mean_ov = float(overlap_rates.mean()); mean_ch = float(chance_rates.mean())
diffs = overlap_rates - chance_rates
se = diffs.std(ddof=1) / np.sqrt(len(diffs))
z = float(diffs.mean() / (se + 1e-12))
from scipy.stats import norm as _norm
p = float(2 * (1 - _norm.cdf(abs(z))))
print(f'Mean overlap: {mean_ov*100:.2f}%  chance(5/N): {mean_ch*100:.2f}%  paired z = {z:.2f}, p = {p:.2e}')

out['overlap_mean'] = mean_ov
out['overlap_std'] = float(overlap_rates.std())
out['chance_5N_mean'] = mean_ch
out['paired_z'] = z
out['paired_p'] = p
'''
anchor = "out = {'N_attention': total"
src = src.replace(anchor, b3 + "\n" + anchor)

open('m7_carbon_donors_smiles.py', 'w', encoding='utf-8').write(src)
print('built m7_carbon_donors_smiles.py')

# syntax check
import ast
ast.parse(src)
print('syntax OK')
