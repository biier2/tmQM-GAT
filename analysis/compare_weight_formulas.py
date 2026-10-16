"""对比不同 metal weighting 公式的权重分布"""
import numpy as np
from collections import Counter
import pandas as pd, gzip

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

def parse_atoms_only(xyz_p):
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
            header = blocks[i].strip(); i += 1
            atoms = []
            for _ in range(n_atoms):
                if i >= len(blocks): break
                cline = blocks[i].strip(); i += 1
                cparts = cline.split()
                if len(cparts) >= 4:
                    z = atom_to_z(cparts[0]); atoms.append(z)
            if any(is_TM(z) for z in atoms) and len(atoms) > 1:
                atoms_list.append(atoms)
    return atoms_list

print("Loading atoms...")
all_atoms = []
for part in [1, 2, 3]:
    all_atoms.extend(parse_atoms_only(f"data_tmQM/local_vB/tmQM_X{part}.xyz.gz"))
N_SAMPLE = 30000
rng = np.random.RandomState(42)
perm = rng.permutation(len(all_atoms))[:N_SAMPLE]
sampled_atoms = [all_atoms[i] for i in perm]
total_c = len(sampled_atoms)

mc = Counter()
for atoms in sampled_atoms:
    for z in set(atoms):
        if z >= 21: mc[z] += 1

n_metal_types = len(mc)  # K
print(f"Metal types: {n_metal_types}, Total samples: {total_c}")

# ===== 试 4 种公式 =====
formulas = {}

# 1. 当前公式 (退化)
w_map1 = {z: min(5.0, total_c/(c+1)) for z, c in mc.items()}

# 2. sqrt (刚才试的)
w_map2 = {}
for z, c in mc.items():
    w_map2[z] = np.clip(np.sqrt(total_c / c), 0.2, 5.0)

# 3. 1/log(n_m+1) scale
w_map3 = {}
for z, c in mc.items():
    w_map3[z] = np.clip(1.0 / np.log(c + 1), 0.2, 5.0)

# 4. N_total/(K*n_m), K=金属种类数
w_map4 = {}
for z, c in mc.items():
    w_map4[z] = np.clip(total_c / (n_metal_types * c), 0.2, 5.0)

# 5. (N_total/n_m)^0.5 * 0.5  (aggressive)
w_map5 = {}
for z, c in mc.items():
    w_map5[z] = np.clip(np.sqrt(total_c / c), 0.5, 3.0)

formulas = [
    ("Current: min(5, N/n)", w_map1),
    ("sqrt(N/n), clamp[0.2,5]", w_map2),
    ("1/log(n+1), clamp[0.2,5]", w_map3),
    ("N/(K*n), clamp[0.2,5]", w_map4),
    ("sqrt(N/n), clamp[0.5,3]", w_map5),
]

print(f"\n{'Formula':<35s} {'min':>6s} {'max':>6s} {'mean':>6s} {'std':>6s} {'max/min':>8s} {'n_unique':>10s}")
print("-" * 80)

for name, w_map in formulas:
    weights = []
    for atoms in sampled_atoms:
        w = max((w_map.get(z, 1.0) for z in set(atoms) if z >= 21), default=1.0)
        weights.append(w)
    weights = np.array(weights)
    weights = weights / weights.mean()  # normalize to mean 1
    n_unique = len(np.unique(np.round(weights, 2)))
    print(f"{name:<35s} {weights.min():6.3f} {weights.max():6.3f} {weights.mean():6.3f} {weights.std():6.3f} {weights.max()/weights.min():8.2f} {n_unique:>10d}")

print("\n推荐：公式 2 (sqrt)，因为它产生适中权重范围 (~1.6×)，数值稳定。")
print("也可以在训练中验证哪种公式对稀有金属提升最大。")
