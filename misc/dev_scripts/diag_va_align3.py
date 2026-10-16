# -*- coding: utf-8 -*-
"""诊断3: 348 个 kept_pos<0 真值行的 xyz 块原始元素构成 + 全 xyz 金属口径计数
嫌疑: atom_to_z 表缺镧系 → 镧系配合物块被判"无 28-TM" → 论文保留它们但本地保留集没有。
论文金属口径可能是 z>=21 (含镧系) 而非 28 元素表。
另查 2 处行序跳变 (k=3591, k=8536) 附近的行特征。
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter

SYMBOLS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
Z_OF = {s: i + 1 for i, s in enumerate(SYMBOLS)}
TM28 = [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80]

df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
codes = df['CSD_code'].tolist()
years = df['CSD_years'].tolist()

# ===== 扫 xyz: code -> 元素集合 =====
blk_elems = {}
for part in [1, 2, 3]:
    with gzip.open(f'data_tmQM/local_vB/tmQM_X{part}.xyz.gz', 'rt') as f:
        it = iter(f)
        while True:
            try:
                ln = next(it).strip()
            except StopIteration:
                break
            if not ln or not ln.isdigit():
                continue
            n_at = int(ln)
            title = next(it).strip()
            code = title.split('CSD_code = ')[1].split('|')[0].strip() if 'CSD_code = ' in title else None
            syms = []
            for _ in range(n_at):
                ln2 = next(it, None)
                if ln2 is None: break
                cparts = ln2.strip().split()
                if len(cparts) >= 4:
                    syms.append(cparts[0])
            if code is not None:
                blk_elems[code] = syms
print(f'xyz 块总数: {len(blk_elems):,}', flush=True)

# ===== 全 xyz 金属口径计数 =====
n_tm28 = sum(1 for s in blk_elems.values() if any(Z_OF.get(x, 0) in TM28 for x in s) and len(s) > 1)
n_z21 = sum(1 for s in blk_elems.values() if any(Z_OF.get(x, 0) >= 21 for x in s) and len(s) > 1)
n_lan = sum(1 for s in blk_elems.values() if any(57 <= Z_OF.get(x, 0) <= 71 for x in s))
lan_codes = [c for c, s in blk_elems.items() if any(57 <= Z_OF.get(x, 0) <= 71 for x in s)]
print(f'28表TM块: {n_tm28:,}   z>=21块: {n_z21:,}   含镧系块: {n_lan:,}', flush=True)
print(f'镧系块样例 code: {lan_codes[:10]}', flush=True)

# ===== 348 行 (kept_pos<0) 的块元素 =====
d2 = np.load('diag_va_align2.npz')
kept_pos = d2['kept_pos']; csv_sorted = d2['csv_sorted']
sel = csv_sorted[kept_pos < 0]
print(f'\nkept_pos<0 真值行: {len(sel)}', flush=True)
elem_count = Counter()
no_block = 0
examples = []
for c in sel:
    s = blk_elems.get(str(codes[c]))
    if s is None:
        no_block += 1
        continue
    for x in s:
        elem_count[x] += 1
    if len(examples) < 8:
        examples.append((codes[c], Counter(s)))
print(f'无块 code: {no_block}', flush=True)
print(f'元素计数 (前20): {elem_count.most_common(20)}', flush=True)
print('样例块元素:', flush=True)
for cd, cnt in examples:
    print(f'  {cd}: {dict(cnt)}', flush=True)
z21_in = sum(1 for c in sel if any(Z_OF.get(x, 0) >= 21 for x in blk_elems.get(str(codes[c]), [])))
tm28_in = sum(1 for c in sel if any(Z_OF.get(x, 0) in TM28 for x in blk_elems.get(str(codes[c]), [])))
lan_in = sum(1 for c in sel if any(57 <= Z_OF.get(x, 0) <= 71 for x in blk_elems.get(str(codes[c]), [])))
print(f'\n348 行中: 块含 z>=21 的 {z21_in}   块含 28表TM 的 {tm28_in}   块含镧系的 {lan_in}', flush=True)

# ===== 2 处跳变附近行特征 =====
print('\n跳变点附近行特征 (k=3591: 34967->27345):', flush=True)
for c in [34967, 34966, 34968, 27345, 27344, 27346, 27350, 27400, 27450]:
    print(f'  csv {c}: code={codes[c]} years={years[c]}', flush=True)
print('跳变点附近 (k=8536: 63380->61386):', flush=True)
for c in [63380, 63379, 63381, 61386, 61385, 61387, 61390]:
    print(f'  csv {c}: code={codes[c]} years={years[c]}', flush=True)
