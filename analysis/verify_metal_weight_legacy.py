"""LEGACY: 验证旧 ablation 公式（30k XYZ 抽样）是否退化 — 保留作历史记录。

This is the original script used during the ablation development. It parses the
public tmQM XYZ archives under ``data_tmQM/local_vB/`` and evaluates the
historical formula ``w = max_m min(5.0, N_total/(n_m+1))`` on a 30,000-complex
subsample, i.e. the run behind the earlier ablation. The manuscript's main
cohort uses ``sqrt(N/n_m)`` clipped to [0.5, 3.0]; that cohort is verified by
``analysis/verify_metal_weight.py``.
"""
import numpy as np
from collections import Counter

# 复用与 ablation_full.ipynb 相同的数据加载逻辑
import pandas as pd
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')

def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,"Pd":46,
           "Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,"Mo":42,"W":74,
           "Ir":77,"Os":76,"Re":75}
    return tbl.get(sym,0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80]

# Parse XYZ files to get atom lists (lightweight, no BO needed)
import gzip

def parse_atoms_only(xyz_p):
    """只提取原子序数列表，不解析坐标和BO"""
    atoms_list = []
    with gzip.open(xyz_p, "rt") as f:
        blocks = f.read().strip().split(chr(10))
    i = 0
    while i < len(blocks):
        line = blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(blocks): break
            header = blocks[i].strip(); i += 1  # comment line
            atoms = []
            for _ in range(n_atoms):
                if i >= len(blocks): break
                cline = blocks[i].strip(); i += 1
                cparts = cline.split()
                if len(cparts) >= 4:
                    z = atom_to_z(cparts[0])
                    atoms.append(z)
            if any(is_TM(z) for z in atoms) and len(atoms) > 1:
                atoms_list.append(atoms)
    return atoms_list

print("Parsing atom lists from XYZ files (no BO needed)...")
all_atoms = []
for part in [1, 2, 3]:
    atoms = parse_atoms_only(f"data_tmQM/local_vB/tmQM_X{part}.xyz.gz")
    all_atoms.extend(atoms)
    print(f"  Part {part}: {len(atoms)} TM complexes")

print(f"\nTotal: {len(all_atoms)} complexes")

# Apply same 30k subsample as ablation notebook
N_SAMPLE = 30000
rng = np.random.RandomState(42)
perm = rng.permutation(len(all_atoms))[:N_SAMPLE]
sampled_atoms = [all_atoms[i] for i in perm]

# ============================================================
# 当前公式 (与 ablation_full.ipynb 完全一致)
# ============================================================
print("\n" + "="*60)
print("CURRENT FORMULA: w = max_m min(5.0, N_total/(n_m+1))")
print("="*60)

mc = Counter()
for atoms in sampled_atoms:
    for z in set(atoms):
        if z >= 21:
            mc[z] += 1

total_c = len(sampled_atoms)
weight_map_current = {z: min(5.0, total_c/(c+1)) for z, c in mc.items()}

# 统计 weight 分布
weights_current = []
for atoms in sampled_atoms:
    w = max((weight_map_current.get(z, 1.0) for z in set(atoms) if z >= 21), default=1.0)
    weights_current.append(w)
weights_current = np.array(weights_current)

print(f"\nMetal counts (top 10):")
for z, c in sorted(mc.items(), key=lambda x: -x[1])[:10]:
    sym = {24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',
           42:'Mo',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',78:'Pt',79:'Au'}.get(z, str(z))
    raw_w = total_c/(c+1)
    print(f"  Z={z:2d} ({sym:3s}): count={c:6d} ({c/total_c*100:5.1f}%), N_total/n = {raw_w:.1f}, weight = {weight_map_current[z]:.1f}")

print(f"\nWeight distribution (current formula):")
print(f"  min={weights_current.min():.3f}, max={weights_current.max():.3f}")
print(f"  mean={weights_current.mean():.3f}, median={np.median(weights_current):.3f}")
print(f"  unique values: {np.unique(weights_current)}")
print(f"  fraction = 5.0: {(weights_current == 5.0).mean()*100:.1f}%")

if len(np.unique(weights_current)) <= 1:
    print("\n  *** DEGENERATE: All weights are identical! Formula fails. ***")
elif (weights_current == 5.0).mean() > 0.95:
    print("\n  *** NEARLY DEGENERATE: >95% weights clipped to 5.0. Formula barely works. ***")
else:
    print("\n  Formula produces meaningful variation.")

# ============================================================
# 修正公式建议
# ============================================================
print("\n" + "="*60)
print("PROPOSED FIX: w_m = sqrt(N_total / n_m), clamp [0.2, 5.0], normalize mean=1")
print("="*60)

weight_map_fixed = {}
for z, c in mc.items():
    raw = np.sqrt(total_c / c)
    weight_map_fixed[z] = np.clip(raw, 0.2, 5.0)

weights_fixed = []
for atoms in sampled_atoms:
    w = max((weight_map_fixed.get(z, 1.0) for z in set(atoms) if z >= 21), default=1.0)
    weights_fixed.append(w)
weights_fixed = np.array(weights_fixed)
# Normalize to mean 1
weights_fixed = weights_fixed / weights_fixed.mean()

print(f"\nPer-metal weights (fixed formula):")
for z, c in sorted(mc.items(), key=lambda x: -x[1])[:10]:
    sym = {24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',
           42:'Mo',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',78:'Pt',79:'Au'}.get(z, str(z))
    print(f"  Z={z:2d} ({sym:3s}): count={c:6d} ({c/total_c*100:5.1f}%), weight={weight_map_fixed[z]:.3f}")

print(f"\nWeight distribution (fixed formula, after mean normalization):")
print(f"  min={weights_fixed.min():.3f}, max={weights_fixed.max():.3f}")
print(f"  mean={weights_fixed.mean():.3f}, std={weights_fixed.std():.3f}")
print(f"  unique values: {len(np.unique(np.round(weights_fixed, 2)))}")
print(f"  % with weight > 2.0: {(weights_fixed > 2.0).mean()*100:.1f}%")
print(f"  % with weight < 0.5: {(weights_fixed < 0.5).mean()*100:.1f}%")

print("\nDone.")
