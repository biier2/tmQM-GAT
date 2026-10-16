# -*- coding: utf-8 -*-
"""意见2(b): 新域测试集 ETKDG+MMFF 部署几何图 (与 scorer_3d 协议一致: AddHs→Embed(42,200)→MMFF→RemoveHs)
节点/边拓扑直接复用 new_domain_iso.pt (官方 XYZ 版), 仅 RBF 距离来自 ETKDG 构象; BO=0, 节点BO3=0
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.logger().setLevel(RDLogger.ERROR)

DATA = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
te_idx = DATA['te_idx']
csv_idx = DATA['csv_idx']
y_csv = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
SMILES_LIST = y_csv['SMILES'].tolist()
rbf_c = np.linspace(1.0, 5.0, 16).astype(np.float32)
rbf_sigma = 0.2

gB, gD, ok = [], [], []
for t, ti in enumerate(te_idx):
    smi = SMILES_LIST[int(csv_idx[ti])]
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if mol is None:
        ok.append(0); gB.append(None); gD.append(None); continue
    try:
        mh = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mh, randomSeed=42, maxAttempts=200) != 0:
            ok.append(0); gB.append(None); gD.append(None); continue
        AllChem.MMFFOptimizeMolecule(mh)
    except Exception:
        ok.append(0); gB.append(None); gD.append(None); continue
    mol2 = Chem.RemoveHs(mh)
    conf = mol2.GetConformer()
    n8 = torch.as_tensor(DATA['node8s'][ti], dtype=torch.float32)
    ei = DATA['ei_ls'][ti]
    e6 = DATA['e6_ls'][ti]
    E = e6.size(0)
    rbf = torch.zeros(E, 16)
    for k in range(E):
        i, j = int(ei[0, k]), int(ei[1, k])
        d = float(np.linalg.norm(np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j))))
        rbf[k] = torch.from_numpy(np.exp(-(d - rbf_c) ** 2 / rbf_sigma).astype(np.float32))
    gB.append({'x': n8, 'edge_index': ei, 'edge_attr': torch.cat([e6, rbf], 1)})
    gD.append({'x': torch.cat([n8, torch.zeros(n8.size(0), 3)], 1),
               'edge_index': ei, 'edge_attr': torch.cat([e6, rbf, torch.zeros(E, 1)], 1)})
    ok.append(1)
    if (t + 1) % 2000 == 0:
        print(f'  {t+1}/{len(te_idx)}, ok={sum(ok)}', flush=True)

torch.save({'graphs_B': gB, 'graphs_D': gD, 'ok': np.array(ok, dtype=bool),
            'te_idx': te_idx}, 'etkdg_test_graphs.pt')
print(f'ETKDG 成功率: {sum(ok)}/{len(ok)} = {sum(ok)/len(ok)*100:.1f}%')
