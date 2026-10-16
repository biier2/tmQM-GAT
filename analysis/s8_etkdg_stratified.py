# -*- coding: utf-8 -*-
"""S8: ETKDG+MMFF 成功率按金属族/金属×配位数分层（全量 tmQM SMILES，multiprocessing）
输出 s8_etkdg_stratified.json + markdown 表。不涉及任何模型训练。"""
import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from multiprocessing import Pool
from collections import Counter
import os

TM_METALS = set([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                 72, 73, 74, 75, 76, 77, 78, 79, 80])
METAL_SYMBOLS = {21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
                 29: 'Cu', 30: 'Zn', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru',
                 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re',
                 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg'}

df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
smiles_list = df['SMILES'].tolist()
print(f'Total SMILES: {len(smiles_list):,}')


def try_etkdg(args):
    idx, smi = args
    if not isinstance(smi, str) or smi == '':
        return (idx, None, None, 'invalid')
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return (idx, None, None, 'parse_failed')
    metals = [(a.GetAtomicNum(), a.GetDegree()) for a in mol.GetAtoms()
              if a.GetAtomicNum() in TM_METALS]
    if not metals:
        return (idx, None, None, 'no_tm')
    z, cn = metals[0]  # 第一个 TM 金属及其配位数（重原子连接数）
    mol = Chem.AddHs(mol)
    try:
        st = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200)
        if st != 0:
            return (idx, z, cn, 'embed_failed')
        AllChem.MMFFOptimizeMolecule(mol)
        return (idx, z, cn, 'success')
    except Exception:
        return (idx, z, cn, 'mmff_crashed')


if __name__ == '__main__':
    n_workers = min(16, os.cpu_count() or 4)
    print(f'Workers: {n_workers}')
    pool = Pool(n_workers)
    results = pool.map(try_etkdg, enumerate(smiles_list), chunksize=256)
    pool.close(); pool.join()

    def cn_bucket(c):
        return c if c < 7 else 7  # 7 = 7+

    stats = {}   # (group, symbol, cn) -> Counter(status)
    n_success = n_valid = 0
    for idx, z, cn, status in results:
        if z is None:
            continue
        n_valid += 1
        if status == 'success':
            n_success += 1
        g = '3d' if z <= 30 else ('4d' if z <= 48 else '5d')
        stats.setdefault((g, METAL_SYMBOLS[z], cn_bucket(cn)), Counter())[status] += 1

    print(f'\nValid TM complexes: {n_valid:,}, success: {n_success:,} ({n_success / n_valid * 100:.1f}%)')

    out = {'n_valid': n_valid, 'n_success': n_success,
           'success_rate_pct': round(n_success / n_valid * 100, 2)}
    # 按金属族 × 配位数桶
    group_cn = {}
    for (g, sym, cn), c in stats.items():
        key = (g, cn)
        group_cn.setdefault(key, Counter())
        group_cn[key] += c
    print('\n=== Success rate by metal group x coordination number ===')
    print(f'{"Group":>4s} {"CN":>4s} {"N":>7s} {"Success":>7s} {"Rate":>7s}')
    out['by_group_cn'] = {}
    for g in ['3d', '4d', '5d']:
        for cn in sorted(set(c for (gg, c) in group_cn if gg == g)):
            c = group_cn.get((g, cn), Counter())
            n = c['success'] + c['embed_failed'] + c['mmff_crashed'] + c['no_tm']
            s = c['success']
            rate = s / n * 100 if n else 0.0
            out['by_group_cn'][f'{g}-CN{cn}'] = {'N': n, 'success': s,
                                                  'embed_failed': c['embed_failed'],
                                                  'mmff_crashed': c['mmff_crashed'],
                                                  'rate_pct': round(rate, 1)}
            print(f'{g:>4s} {cn:>4d} {n:>7d} {s:>7d} {rate:6.1f}%')

    # 按金属
    out['by_metal'] = {}
    print('\n=== Success rate by metal (N>=100) ===')
    for g in ['3d', '4d', '5d']:
        for sym in sorted(set(s for (gg, s, c) in stats if gg == g)):
            c = Counter()
            for (gg, s2, cn), cc in stats.items():
                if gg == g and s2 == sym:
                    c += cc
            n = c['success'] + c['embed_failed'] + c['mmff_crashed'] + c['no_tm']
            if n < 100:
                continue
            s = c['success']
            rate = s / n * 100 if n else 0.0
            out['by_metal'][sym] = {'N': n, 'success': s, 'rate_pct': round(rate, 1)}
            print(f'{g} {sym:>3s} {n:>7d} {s:>7d} {rate:6.1f}%')

    json.dump(out, open('s8_etkdg_stratified.json', 'w'), indent=1)
    print('\nsaved s8_etkdg_stratified.json')
