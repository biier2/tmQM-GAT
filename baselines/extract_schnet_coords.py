# -*- coding: utf-8 -*-
"""
从 XYZ 提取 new_domain_iso.pt 那 90,377 个络合物的**正确**坐标。
关键：使用 rebuild_match_iso.py 的全局块索引（三个 part 顺序拼接，bidx = CSV 行号），
而不是 schNet/DimeNet 旧脚本里 start_idx=len(all_m) 的那个 bug。
产出 schnet_domain_coords.npz：coords(flat) + lens，与 new_domain_iso.pt 的 csv_idx 顺序一一对应。
"""
import sys, gzip, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch

ROOT = 'data_tmQM/local_vB'

def parse_xyz(path):
    blocks = []
    with gzip.open(path, 'rt') as f:
        lines = f.read().strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line: continue
        if line.isdigit():
            n = int(line)
            comment = lines[i].strip(); i += 1
            code = comment.split('=')[1].split('|')[0].strip() if 'CSD_code' in comment else '?'
            syms, coords = [], []
            for _ in range(n):
                if i >= len(lines): break
                c = lines[i].strip(); i += 1
                cp = c.split()
                if len(cp) >= 4:
                    try:
                        syms.append(cp[0]); coords.append([float(x) for x in cp[1:4]])
                    except ValueError: pass
            blocks.append((code, syms, np.array(coords, dtype=np.float32)))
    return blocks

t0 = time.time()
xyz_blocks = []
for p in [1, 2, 3]:
    xyz_blocks.extend(parse_xyz(f'{ROOT}/tmQM_X{p}.xyz.gz'))
    print(f'  part {p}: cum {len(xyz_blocks)}  [{time.time()-t0:.0f}s]', flush=True)
print(f'XYZ 总块数: {len(xyz_blocks)}')

D = torch.load('new_domain_iso.pt', weights_only=False)
csv_idx = np.asarray(D['csv_idx'])
atoms_filt = D['atoms_filt']
N = len(csv_idx)
print(f'new_domain_iso: N={N}')

PT = None
from rdkit.Chem import GetPeriodicTable
PT = GetPeriodicTable()

coords_out = []
mismatch = 0
for k in range(N):
    ci = int(csv_idx[k])
    code, syms, coords = xyz_blocks[ci]
    zs = np.array([PT.GetAtomicNumber(s) for s in syms], dtype=np.int64)
    if len(coords) != len(atoms_filt[k]) or not np.array_equal(zs, np.asarray(atoms_filt[k])):
        mismatch += 1
        if mismatch <= 3:
            print(f'  [!] k={k} csv_idx={ci}: coords {len(coords)} vs filt {len(atoms_filt[k])}')
    coords_out.append(coords)
    if (k + 1) % 20000 == 0:
        print(f'  ...{k+1}/{N} [{time.time()-t0:.0f}s]', flush=True)

print(f'元素/长度不一致: {mismatch} / {N}  ' + ('✓ 全部对齐' if mismatch == 0 else '❌'))
lens = np.array([len(c) for c in coords_out], dtype=np.int32)
flat = np.concatenate(coords_out, axis=0)
np.savez_compressed('schnet_domain_coords.npz', coords=flat, lens=lens)
print(f'SAVED schnet_domain_coords.npz: {flat.shape} 原子, {lens.sum()} 校验和, {flat.nbytes/1e6:.1f} MB')
print(f'总耗时 {time.time()-t0:.0f}s')
