# -*- coding: utf-8 -*-
"""id12b · 双栈全量 SMILES 解析（同一 CSV，单个 rdkit 版本跑一遍）
用法: python dual_parse_one.py <tag>     # tag ∈ base2026 | rd2024
输出: _recon_20261016\parse_<tag>.json + parse_<tag>.npz
字段: status(0 ok/1 empty/2 parse_none/3 sanitize_err/4 exc), tm29, tm30(=tm29+La), natoms, la
"""
import sys, os, json, time
import numpy as np, pandas as pd
import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

CSV = r'D:\ml\chem_ml5\data_tmQM\local_vB\tmQM_y.csv'
OUT = r'D:\ml\paper\draft\round8\_recon_20261016'
TAG = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

TM29 = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        72, 73, 74, 75, 76, 77, 78, 79, 80]
TM30 = TM29 + [57]   # + La
S29, S30 = set(TM29), set(TM30)

df = pd.read_csv(CSV, sep=';')
smi_list = df['SMILES'].tolist()
codes = df['CSD_code'].tolist()
n = len(smi_list)
t0 = time.time()
print(f'[{TAG}] rdkit {rdkit.__version__} | py {sys.version.split()[0]} | CSV rows: {n:,}', flush=True)

ST_OK, ST_EMPTY, ST_PARSE_NONE, ST_SANITIZE, ST_EXC = 0, 1, 2, 3, 4
status = np.zeros(n, np.int8)
tm29 = np.zeros(n, bool)
tm30 = np.zeros(n, bool)
la = np.zeros(n, bool)
natoms = np.zeros(n, np.int32)
failures = {}   # idx -> [code, msg]

for i, s in enumerate(smi_list):
    if not isinstance(s, str) or s.strip() == '':
        status[i] = ST_EMPTY
        failures[i] = [ST_EMPTY, 'empty or nan']
        continue
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        msg = ''
        try:
            m2 = Chem.MolFromSmiles(s, sanitize=False)
            if m2 is None:
                status[i] = ST_PARSE_NONE
                msg = 'MolFromSmiles(sanitize=False) -> None'
            else:
                try:
                    Chem.SanitizeMol(m2)
                    status[i] = ST_PARSE_NONE
                    msg = 'unexpected: sanitize OK on retry'
                except Exception as e:
                    status[i] = ST_SANITIZE
                    msg = f'{type(e).__name__}: {e}'
        except Exception as e:
            status[i] = ST_EXC
            msg = f'{type(e).__name__}: {e}'
        failures[i] = [int(status[i]), msg[:300]]
        continue
    status[i] = ST_OK
    za = {a.GetAtomicNum() for a in mol.GetAtoms()}
    natoms[i] = mol.GetNumAtoms()
    tm29[i] = any(z in S29 for z in za)
    tm30[i] = any(z in S30 for z in za)
    la[i] = 57 in za
    if (i + 1) % 20000 == 0:
        print(f'  {i+1:,} rows, {time.time() - t0:.0f}s', flush=True)

n_ok = int((status == ST_OK).sum())
n_empty = int((status == ST_EMPTY).sum())
n_fail = n - n_ok
n_v29 = int((tm29).sum())
n_v30 = int((tm30).sum())
n_la = int(la.sum())
by_st = {int(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))}
meta = {
    'tag': TAG, 'rdkit': rdkit.__version__, 'python': sys.version.split()[0],
    'csv': CSV, 'n_rows': n,
    'n_parse_ok': n_ok, 'n_fail': n_fail, 'n_empty': n_empty,
    'n_valid_tm29': n_v29, 'n_valid_tm30': n_v30, 'n_mols_with_La': n_la,
    'status_counts': {str(k): v for k, v in by_st.items()},
    'n_failures_recorded': len(failures),
    'elapsed_sec': round(time.time() - t0, 1),
}
with open(os.path.join(OUT, f'parse_{TAG}.json'), 'w', encoding='utf-8') as f:
    json.dump({'meta': meta, 'failures': {str(k): v for k, v in failures.items()}}, f, ensure_ascii=False)
np.savez_compressed(os.path.join(OUT, f'parse_{TAG}.npz'),
                    status=status, tm29=tm29, tm30=tm30, la=la, natoms=natoms)

print(f'[{TAG}] parse_ok={n_ok:,}  fail={n_fail:,} (empty={n_empty})', flush=True)
print(f'[{TAG}] valid TM29 = {n_v29:,} | TM30(+La) = {n_v30:,} | mols w/ La = {n_la:,}', flush=True)
print(f'[{TAG}] done in {time.time() - t0:.0f}s', flush=True)
