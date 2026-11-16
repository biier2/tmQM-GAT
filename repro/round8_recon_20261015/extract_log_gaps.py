# -*- coding: utf-8 -*-
"""Read-only: extract from the Gaussian logs in D:\\ml\\log the final SP job's
Alpha/Beta HOMO-LUMO gap (eV), charge, multiplicity and route line.
Output log_gaps.txt (for reconciling comment 43 / Table S16).

Gaussian Pop=Full output layout (the prefix is repeated on every line and continuation line):
  Alpha occ. block (low energy→HOMO) → Alpha virt. block (LUMO→high energy)
  [for UHF additionally] Beta occ. block → Beta virt. block
Take the "last printed": the last value of the last Alpha-occ line = HOMO(α);
the first value of the first Alpha-virt line after it = LUMO(α); likewise for Beta.
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

LOGDIR = r'D:\ml\log'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_gaps.txt')
HA_EV = 27.211386245988

lines_out = []
for fn in sorted(os.listdir(LOGDIR)):
    if not fn.endswith('.log'):
        continue
    txt = open(os.path.join(LOGDIR, fn), 'r', encoding='utf-8', errors='replace').read()

    seq = []  # (pos, spin, kind, vals) in order of appearance
    for m in re.finditer(r'^\s*(Alpha|Beta)\s+(occ|virt)\.\s+eigenvalues\s+--(.*)$',
                         txt, flags=re.M):
        vals = [float(x) for x in re.findall(r'-?\d+\.\d{5}', m.group(3))]
        if vals:
            seq.append((m.start(), m.group(1), m.group(2), vals))

    mults = re.findall(r'Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)', txt)
    routes = re.findall(r'^\s*#p?\s*(.+)$', txt, flags=re.M)

    lines_out.append('===== %s =====' % fn)
    lines_out.append('  Charge/Mult: %s' % sorted(set(mults)))
    for i, rt in enumerate(routes):
        lines_out.append('  route%d: # %s' % (i + 1, rt.strip()[:120]))

    def block(spin, kind):
        return [e for e in seq if e[1] == spin and e[2] == kind]

    a_occ, a_virt = block('Alpha', 'occ'), block('Alpha', 'virt')
    b_occ, b_virt = block('Beta', 'occ'), block('Beta', 'virt')

    gaps = {}
    if a_occ and a_virt:
        homo_a = a_occ[-1][3][-1]
        after = [e for e in a_virt if e[0] > a_occ[-1][0]]
        if after:
            lumo_a = after[0][3][0]
            gaps['alpha'] = (homo_a, lumo_a, (lumo_a - homo_a) * HA_EV)
    if b_occ and b_virt:
        homo_b = b_occ[-1][3][-1]
        after = [e for e in b_virt if e[0] > b_occ[-1][0]]
        if after:
            lumo_b = after[0][3][0]
            gaps['beta'] = (homo_b, lumo_b, (lumo_b - homo_b) * HA_EV)

    for spin in ('alpha', 'beta'):
        if spin in gaps:
            h, l, g = gaps[spin]
            lines_out.append('  %s HOMO=%.5f LUMO=%.5f -> gap=%.4f eV' % (spin, h, l, g))
    if gaps:
        gmin = min(v[2] for v in gaps.values())
        lines_out.append('  >>> reported gap (smaller one for open shell) = %.4f eV' % gmin)
    lines_out.append('')

open(OUT, 'w', encoding='utf-8').write('\n'.join(lines_out))
print('\n'.join(lines_out))
print('-> %s' % OUT)
