# -*- coding: utf-8 -*-
"""Read-only three-part check:
(1) tmQMg CSV header + COLUMN_MAP target fields (including the zeta-level issue for E_elec);
(2) RDKit EmbedMolecule convenience-function defaults (ETKDG version / knowledge terms);
(3) s1_enrichment_all.json full-table EF (aligned to the printed definition of table S15: paper = nominal × fN/K),
   and comparison against the SI printed values at the 1% threshold to determine the tail direction actually used for each property."""
import csv
import json
import math
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ---------- (1) tmQMg fields ----------
CSV = r'D:\ml\chem_ml5\tmQMg_properties_and_targets.csv'
print('===== ① tmQMg CSV =====')
with open(CSV, encoding='utf-8', errors='replace', newline='') as f:
    rd = csv.reader(f)
    header = next(rd)
    first = next(rd)
print('n_cols =', len(header))
MAP = {
    'gap': 'tzvp_homo_lumo_gap',
    'homo': 'tzvp_homo_energy',
    'lumo': 'tzvp_lumo_energy',
    'dipole': 'tzvp_dipole_moment',
    'E_elec': 'tzvp_electronic_energy',
    'E_disp': 'tzvp_dispersion_energy',
    'polarizability': 'polarisability',
}
print('-- Presence of COLUMN_MAP fields in the header and sample values --')
for k, col in MAP.items():
    ok = col in header
    idx = header.index(col) if ok else -1
    print('  %-15s -> %-28s %s  sample=%s' % (k, col, 'OK' if ok else 'MISSING',
          first[idx] if ok else '-'))
print('-- All columns containing energy / gap / homo / lumo / dipole / polaris --')
for i, h in enumerate(header):
    hl = h.lower()
    if any(t in hl for t in ('energy', 'gap', 'homo', 'lumo', 'dipole', 'polaris')):
        print('   [%2d] %s = %s' % (i, h, first[i]))

# ---------- (2) RDKit EmbedMolecule defaults ----------
print('\n===== (2) RDKit EmbedMolecule convenience function =====')
import os
import re
import rdkit
print('rdkit version =', rdkit.__version__)
allchem = os.path.join(os.path.dirname(rdkit.__file__), 'Chem', 'AllChem.py')
src = open(allchem, encoding='utf-8', errors='replace').read()
m = re.search(r'def EmbedMolecule\(', src)
if m:
    body = src[m.start():m.start() + 3500]
    # print the fragments of the function body that involve parameter setting
    for ln in body.splitlines():
        if any(t in ln for t in ('params', 'TorsionAngle', 'BasicKnowledge', 'ETversion',
                                 'useRandomCoords', 'EmbedParameters', 'maxAttempts', 'randomSeed',
                                 'def EmbedMolecule')):
            print('  |', ln.rstrip()[:150])
from rdkit.Chem import rdDistGeom
p = rdDistGeom.EmbedParameters()
print('-- EmbedParameters() construction defaults --')
for a in ('ETversion', 'useExpTorsionAnglePrefs', 'useBasicKnowledge', 'useSmallRingTorsions',
          'useMacrocycleTorsions', 'useMacrocycle14config', 'enforceChirality',
          'useRandomCoords', 'maxAttempts', 'randomSeed', 'pruneRmsThresh', 'numThreads'):
    try:
        print('   %-26s = %s' % (a, getattr(p, a)))
    except AttributeError as e:
        print('   %-26s = <n/a>' % a)

# ---------- (3) EF full table ----------
print('\n===== (3) s1_enrichment_all.json =====')
J = r'D:\ml\chem_ml5\s1_enrichment_all.json'
d = json.load(open(J, encoding='utf-8'))
# flat keys: 'family|prop|dir|frac'
rows = {}
fams = set()
props = set()
dirs = set()
for k, node in d.items():
    fam, prop, dr, frac = k.split('|')
    fams.add(fam)
    props.add(prop)
    dirs.add(dr)
    rows[(fam, prop, dr, float(frac))] = node
print('families =', sorted(fams))
print('props    =', sorted(props))
print('dirs     =', sorted(dirs))

N = 13556
SI1 = {  # 1% printed values: A / composition
    'HL-Gap': (46.2, 5.1), 'HOMO': (78.4, 8.8), 'LUMO': (69.6, 2.9), 'Dipole': (56.4, 4.4),
    'MetalCharge': (94.5, 35.2), 'E_elec': (90.1, 99.7), 'E_disp': (87.9, 85.0),
    'Polarizability': (81.4, 90.9),
}
SI_ALIAS = {
    'gap': 'HL-Gap', 'homo': 'HOMO', 'lumo': 'LUMO', 'dipole': 'Dipole',
    'mcharge': 'MetalCharge', 'metal_charge': 'MetalCharge', 'metalcharge': 'MetalCharge',
    'E_elec': 'E_elec', 'eelec': 'E_elec', 'e_elec': 'E_elec',
    'E_disp': 'E_disp', 'edisp': 'E_disp', 'e_disp': 'E_disp',
    'polar': 'Polarizability', 'polarizability': 'Polarizability',
    'polarisability': 'Polarizability',
}
print('\n-- 1% matching (paper = ef_json × fN/K), determining the tail direction actually used per property --')
print('%-6s %-13s %-5s %-9s %-9s' % ('fam', 'prop', 'dir', 'paper', 'SI'))
for (fam, prop, dr, f) in sorted(rows):
    if abs(f - 0.01) > 1e-9 or fam not in ('A', 'comp'):
        continue
    node = rows[(fam, prop, dr, f)]
    K = math.ceil(f * N)
    paper = node['ef'] * (f * N) / K
    alias = SI_ALIAS.get(prop, SI_ALIAS.get(prop.lower()))
    if not alias:
        continue
    side = 0 if fam == 'A' else 1
    target = SI1[alias][side]
    ok = 'MATCH' if abs(round(paper, 1) - target) < 0.051 else ''
    print('%-6s %-13s %-5s %-9.2f %-9.1f %s' % (fam, prop, dr, paper, target, ok))

print('\n-- Full extended table (paper definition, 1 decimal place): per property × (low/high) × {1,5,10}% --')
for (fam, prop, dr, f) in sorted(rows):
    if fam not in ('A', 'composition'):
        continue
    node = rows[(fam, prop, dr, f)]
    K = math.ceil(f * N)
    paper = node['ef'] * (f * N) / K
    SI_ALIAS.get(prop)
    # the summary is printed in the second block below
out = {}
for (fam, prop, dr, f) in rows:
    if fam not in ('A', 'comp'):
        continue
    K = math.ceil(f * N)
    out[(fam, prop, dr, f)] = rows[(fam, prop, dr, f)]['ef'] * (f * N) / K
for prop in sorted(props):
    for dr in sorted(dirs):
        line = '%-14s %-5s' % (prop, dr)
        for f in (0.01, 0.05, 0.10):
            try:
                va = out[('A', prop, dr, f)]
                vc = out[('comp', prop, dr, f)]
                line += ' | %5.1f / %5.1f' % (round(va, 1), round(vc, 1))
            except KeyError:
                line += ' |   n/a        '
        print(line)
