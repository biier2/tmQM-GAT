# -*- coding: utf-8 -*-
"""R5: 从 data_tmQM/local_vB/tmQM_X1/X2/X3.xyz.gz 提取自旋态 (header 第二行 S = 字段)"""
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

print(f'总结构: {len(spin_map)}, 解析失败: {n_fail}')
c = Counter(spin_map.values())
for k in sorted(c, key=lambda x: float(x)):
    print(f'  S={k}: {c[k]}')
# 开壳层 = S > 0 (S=0 单重态为闭壳层; S=1/2 双重态起为开壳层)
open_shell = sum(v for k, v in c.items() if float(k) > 0)
print(f'开壳层占比: {open_shell}/{len(spin_map)} = {open_shell/len(spin_map)*100:.2f}%')
