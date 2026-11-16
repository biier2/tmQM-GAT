# -*- coding: utf-8 -*-
"""
Case-study validation analysis: scorer predictions vs DFT (TPSSh-D3BJ/def2-SVP)
Data sources:
  - scorer_3d_results.csv (18 catalysts, 3D GAT + SMILES GAT dual-model predictions)
  - case_study_validation.md (authoritative gap / dipole of the 10 DFT jobs; open-shell uses the smaller alpha/beta gap)
Mapping (electron-count method, see the project handover document §4.1; ⚠ marks entries pending gjf confirmation):
  Co-cat3_sp→Co-cat1, Fe-cat1_sp→Fe-cat1, Ni-cat3_sp→Ni-cat3,
  Pd-cat2_sp→Pd-cat2(⚠), Zr-cat3-2_sp→Zr-cat3(⚠),
  cis-PdCl2-PPh32→cis-PdCl2(PPh3)2, trans-Pd-PPh_sp→trans-PdCl2(PPh3)2,
  Pt-dppe_sp→Pt(dppe)Cl2, Rh-CO2_sp→Rh(CO)2(acac), Ni-NHC_sp→(old 51-atom molecule, deprecated)
(The original optimized Ni-NHC structure = the 51-atom molecule with the C2-H retained; after re-modelling Ni(IMes)2Cl2 m1/m3,
 the singlet m1 is the ground state, log re-checked 2026-08-24, m1 adopted)
"""
import pandas as pd

# DFT reference values: gap (eV, open-shell=min(alpha,beta)), dipole (D)
DFT = {
    # old 5
    'Co-cat1':            {'gap': 3.304110, 'dipole': 0.0016},
    'Fe-cat1':            {'gap': 2.763144, 'dipole': 9.7600},
    'Ni-cat3':            {'gap': 1.393083, 'dipole': 5.6682},
    'Pd-cat2':            {'gap': 2.925023, 'dipole': 2.4821},
    'Zr-cat3':            {'gap': 3.297484, 'dipole': 5.3175},
    # new 5
    'cis-PdCl2(PPh3)2':   {'gap': 3.169948, 'dipole': 10.8571},
    'trans-PdCl2(PPh3)2': {'gap': 3.193102, 'dipole': 0.0322},
    'Rh(CO)2(acac)':      {'gap': 4.027482, 'dipole': 3.1738},
    'Ni(IMes)2Cl2':       {'gap': 3.235281, 'dipole': 0.0003},   # ✓ singlet m1 (log re-checked 2026-08-24)
    'Pt(dppe)Cl2':        {'gap': 3.953475, 'dipole': 10.2881},
}

OLD5 = ['Co-cat1', 'Fe-cat1', 'Ni-cat3', 'Pd-cat2', 'Zr-cat3']
NEW5 = ['cis-PdCl2(PPh3)2', 'trans-PdCl2(PPh3)2', 'Rh(CO)2(acac)', 'Ni(IMes)2Cl2', 'Pt(dppe)Cl2']

df = pd.read_csv('scorer_3d_results.csv')
df['Metal'] = df['Name'].str.extract(r'(Zr|Ni|Pd|Co|Fe|Rh|Pt)')

rows = []
for name in OLD5 + NEW5:
    r = df[df['Name'] == name].iloc[0]
    d = DFT[name]
    rows.append({
        'Catalyst': name,
        'Metal': r['Metal'],
        'Model': r['Model'],
        'gap_3D': r['gap_eV'],      'gap_DFT': d['gap'],      'd_gap': abs(r['gap_eV'] - d['gap']),
        'gap_SM': r['gap_SMILES_eV'], 'd_gap_SM': abs(r['gap_SMILES_eV'] - d['gap']),
        'dip_3D': r['dipole_D'],    'dip_DFT': d['dipole'],   'd_dip': abs(r['dipole_D'] - d['dipole']),
        'dip_SM': r['dipole_SMILES_D'], 'd_dip_SM': abs(r['dipole_SMILES_D'] - d['dipole']),
    })
cmp = pd.DataFrame(rows)
pd.set_option('display.width', 250)
pd.set_option('display.float_format', lambda v: f'{v:.3f}')

for grp, names in [('old 5 (fallback-path candidates)', OLD5), ('new 5 (freshly computed DFT this round)', NEW5)]:
    g = cmp[cmp['Catalyst'].isin(names)]
    print(f'\n===== {grp} =====')
    print(g[['Catalyst', 'gap_3D', 'gap_DFT', 'd_gap', 'gap_SM', 'd_gap_SM',
             'dip_3D', 'dip_DFT', 'd_dip', 'dip_SM', 'd_dip_SM']].to_string(index=False))
    print(f'--- {grp} MAE ---')
    print(f"  gap  3D={g['d_gap'].mean():.3f} eV | SMILES={g['d_gap_SM'].mean():.3f} eV")
    print(f"  dip  3D={g['d_dip'].mean():.3f} D  | SMILES={g['d_dip_SM'].mean():.3f} D")

g_all = cmp
print('\n===== MAE over all 10 =====')
print(f"  gap  3D={g_all['d_gap'].mean():.3f} eV | SMILES={g_all['d_gap_SM'].mean():.3f} eV")
print(f"  dip  3D={g_all['d_dip'].mean():.3f} D  | SMILES={g_all['d_dip_SM'].mean():.3f} D")

# all 10 mappings have been confirmed against the logs (see the project handover document §4.1), so no subset sensitivity analysis is needed
print('\n===== Usage statistics =====')
print(df['Model'].value_counts().to_string())
