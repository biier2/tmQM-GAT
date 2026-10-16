# -*- coding: utf-8 -*-
"""id12b② · 实际运行环境中 EmbedMolecule 的默认参数：
   (a) 文档默认值 dump；(b) 各 ETKDG 预置参数 diff；
   (c) 经验等价性测试：wrapper 默认调用 vs 各预置（同 seed 逐坐标比对）→ 判定 wrapper 实际等价于哪个预置。
用法: python embed_defaults.py <tag>
"""
import sys, os, json, inspect
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDistGeom
RDLogger.DisableLog('rdApp.*')

OUT = r'D:\ml\paper\draft\round8\_recon_20261016'
TAG = sys.argv[1]
CSV = r'D:\ml\chem_ml5\data_tmQM\local_vB\tmQM_y.csv'

res = {'tag': TAG, 'rdkit': rdkit.__version__, 'python': sys.version.split()[0]}

# ---------- (a) 文档 ----------
doc = getattr(AllChem.EmbedMolecule, '__doc__', '')
res['allchem_embedmolecule_doc'] = doc
try:
    res['signature'] = str(inspect.signature(AllChem.EmbedMolecule))
except Exception as e:
    res['signature'] = f'<unavailable: {e}>'

# ---------- (b) 预置参数 dump ----------
PRESETS = ['ETKDG', 'ETKDGv2', 'ETKDGv3', 'srETKDGv3', 'KDG', 'ETDG']
SKIP = {'SetSeed', 'SetCoordMap', 'Assign', 'GetName'}

def dump_ps(ps):
    d = {}
    for a in sorted(dir(ps)):
        if a.startswith('_'):
            continue
        try:
            v = getattr(ps, a)
        except Exception as e:
            d[a] = f'<err {e}>'
            continue
        if callable(v):
            continue
        try:
            json.dumps(v)
            d[a] = v
        except Exception:
            d[a] = str(v)
    return d

psets = {}
for name in PRESETS:
    fn = getattr(AllChem, name, None)
    if fn is None:
        fn = getattr(rdDistGeom, name, None)
    try:
        psets[name] = dump_ps(fn())
    except Exception as e:
        psets[name] = {'<err>': str(e)}
psets['EmbedParametersBase'] = dump_ps(rdDistGeom.EmbedParameters())
res['presets'] = psets

# 与 EmbedParametersBase 的差异表
base = psets['EmbedParametersBase']
diffs = {}
for name in PRESETS:
    if name in psets and isinstance(psets[name], dict) and '<err>' not in psets[name]:
        diffs[name] = {k: v for k, v in psets[name].items() if base.get(k, '<missing>') != v}
res['preset_vs_base_diff'] = diffs

# ---------- (c) 经验等价性测试 ----------
df = pd.read_csv(CSV, sep=';', usecols=['SMILES', 'CSD_code'])
test_mols = []
test_mols.append(('cyclopropane_acid', 'C1CC1C(=O)O'))
test_mols.append(('cyclobutane', 'C1CCC1'))
test_mols.append(('benzene', 'c1ccccc1'))
# 取一个真 tmQM Ni / Fe 样例
for sym, key in [('Ni', 'ni'), ('Fe', 'fe')]:
    hit = df[df['SMILES'].astype(str).str.contains(sym)]
    if len(hit):
        r = hit.iloc[len(hit) // 2]
        test_mols.append((f'tmqm_{key}', str(r['SMILES'])))

def embed_wrapper(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    st = AllChem.EmbedMolecule(m, randomSeed=42, maxAttempts=200)
    if st != 0:
        return None
    return np.array(m.GetConformer().GetPositions())

def embed_preset(name, smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    ps = getattr(AllChem, name)()
    ps.randomSeed = 42
    for attr, val in (('maxAttempts', 200), ('numThreads', 1), ('clearConfs', True)):
        try:
            setattr(ps, attr, val)
        except Exception:
            pass
    cid = rdDistGeom.EmbedMolecule(m, ps)
    if cid < 0:
        return None
    return np.array(m.GetConformer(cid).GetPositions())

equiv = {}
for label, smi in test_mols:
    row = {'smiles': smi}
    try:
        w = embed_wrapper(smi)
        row['wrapper_ok'] = w is not None
        for name in PRESETS:
            try:
                p = embed_preset(name, smi)
                if w is None or p is None:
                    row[name] = None
                else:
                    if w.shape == p.shape:
                        row[name] = float(np.abs(w - p).max())
                    else:
                        row[name] = f'shape {w.shape} vs {p.shape}'
            except Exception as e:
                row[name] = f'<err {type(e).__name__}: {e}>'
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'
    equiv[label] = row
    print(f'[equiv] {label}: {json.dumps(row, ensure_ascii=False)}', flush=True)
res['equivalence_max_abs_dcoord'] = equiv

with open(os.path.join(OUT, f'embed_defaults_{TAG}.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(f'[{TAG}] saved embed_defaults_{TAG}.json', flush=True)
