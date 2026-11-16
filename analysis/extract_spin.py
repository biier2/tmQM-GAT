# -*- coding: utf-8 -*-
"""R5: extract spin states from data_tmQM/local_vB/tmQM_X1/X2/X3.xyz.gz (S = field on the second header line)"""
import gzip, re, json, sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
FILES = ['d:/ml/chem_ml5/data_tmQM/local_vB/tmQM_X1.xyz.gz',
         'd:/ml/chem_ml5/data_tmQM/local_vB/tmQM_X2.xyz.gz',
         'd:/ml/chem_ml5/data_tmQM/local_vB/tmQM_X3.xyz.gz']

spin_map = {}   # CSD_code -> S (str)
charge_map = {} # CSD_code -> q
n_fail = 0
for fn in FILES:
    with gzip.open(fn, 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line.strip())
            header = f.readline()
            if not header:
                break
            m = re.search(r'CSD_code = (\S+)', header)
            s = re.search(r'S = (\S+)', header)
            q = re.search(r'q = ([-+]?\d+)', header)
            if not m or not s:
                n_fail += 1
            else:
                spin_map[m.group(1)] = s.group(1)
                if q:
                    charge_map[m.group(1)] = q.group(1)
            # skip atoms
            for _ in range(n):
                f.readline()

with open('d:/ml/chem_ml5/spin_map.json', 'w') as f:
    json.dump({'spin': spin_map, 'charge': charge_map}, f)

print(f'Total structures: {len(spin_map)}, parse failures: {n_fail}')
c = Counter(spin_map.values())
for k in sorted(c, key=lambda x: float(x)):
    print(f'  S={k}: {c[k]}')
# open-shell = S > 0 (S=0 singlet is closed-shell; S=1/2 doublet and above are open-shell)
open_shell = sum(v for k, v in c.items() if float(k) > 0)
print(f'open-shell fraction: {open_shell}/{len(spin_map)} = {open_shell/len(spin_map)*100:.2f}%')
