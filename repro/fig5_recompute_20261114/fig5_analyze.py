# -*- coding: utf-8 -*-
"""Consolidate the Fig. 5 recomputation into report tables."""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')   # archived result files shipped with this package


def _path(name):
    """Resolve an input file: results/ first, then the script directory."""
    p = os.path.join(RES, name)
    return p if os.path.exists(p) else os.path.join(HERE, name)

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge',
           'E_elec', 'E_disp', 'polarizability']
NAME = {'gap': 'HL-Gap', 'homo': 'HOMO', 'lumo': 'LUMO', 'dipole': 'Dipole',
        'metal_charge': 'Metal q', 'E_elec': 'E_elec', 'E_disp': 'E_disp',
        'polarizability': 'Polarizability'}
PUBLISHED = {'homo': 47.6, 'lumo': 44.1, 'gap': 31.3, 'dipole': 26.1,
             'metal_charge': 30.5, 'E_elec': 14.0, 'E_disp': 23.5,
             'polarizability': 21.5}


def load(p):
    with open(_path(p), encoding='utf-8') as f:
        d = json.load(f)
    # COORD is identical to the A_new set (coordinated donors only); alias it so the
    # per-class tables can be written directly.
    for sect in ('share', 'per_atom_ratio', 'paired_z', 'excess_pp', 'frac_above_null'):
        if sect in d:
            for t in d[sect]:
                d[sect][t]['COORD'] = d[sect][t]['A_new']
    if 'pooled_per_atom_ratio' in d:
        for t in d['pooled_per_atom_ratio']:
            d['pooled_per_atom_ratio'][t]['COORD'] = d['pooled_per_atom_ratio'][t]['A_new']
    return d


def tstat(s):
    """One-sample t of the per-molecule paired z list."""
    if not s or s['n'] < 2 or s['sd'] == 0:
        return np.nan, np.nan
    t = s['mean'] / (s['sd'] / np.sqrt(s['n']))
    return t, s['n']


def main():
    test = load('fig5_test_ensemble_pred.json')
    old = load('fig5_fulldomain_ensemble_pred.json')

    lines = []
    A = lines.append

    A('# Fig. 5 recomputation — consolidated tables\n')
    A(f"test-set run: n={test['meta']['n_complexes_used']}, "
      f"test fraction {100 * test['meta']['test_fraction_of_sample']:.1f}%")
    A(f"full-domain run: n={old['meta']['n_complexes_used']}, "
      f"test fraction {100 * old['meta']['test_fraction_of_sample']:.1f}%\n")

    # ---------- Table 1: published definition, old vs new pool
    A('## T1. Published donor definition (A_old = COORD+REMOTE_H), '
      'full-domain vs test-set pool\n')
    A('| property | published | reproduced (full domain) | **test set** | '
      'repro - published | test - full | test - published |')
    A('|---|---|---|---|---|---|---|')
    diffs, repro = [], []
    for t in TARGETS:
        p = PUBLISHED[t]
        o = old['share'][t]['A_old']['mean'] * 100
        n = test['share'][t]['A_old']['mean'] * 100
        diffs.append(n - o)
        repro.append(o - p)
        A(f'| {NAME[t]} | {p:.1f}% | {o:.1f}% | **{n:.1f}%** | '
          f'{o - p:+.2f} | {n - o:+.2f} | {n - p:+.2f} |')
    A(f'\nreproduction fidelity: mean |repro - published| = '
      f'{np.mean(np.abs(repro)):.2f} pp; max = {np.max(np.abs(repro)):.2f} pp')
    A(f'test set vs full domain: mean |test - full| = '
      f'{np.mean(np.abs(diffs)):.2f} pp; max = {np.max(np.abs(diffs)):.2f} pp\n')

    # ---------- Table 2: three-way decomposition on the test set
    A('## T2. Six-way atom typing, test set (share of total |attribution|, %)\n')
    A('| property | METAL | MBC | **COORD** | REMOTE_H | OTHER | REMOTE_C | '
      'A_old | **A_new** | B_old | **B_new** |')
    A('|---|---|---|---|---|---|---|---|---|---|---|')
    for t in TARGETS:
        s = test['share'][t]
        A(f"| {NAME[t]} | {100 * s['METAL']['mean']:.1f} | "
          f"{100 * s['MBC']['mean']:.1f} | **{100 * s['COORD']['mean']:.1f}** | "
          f"{100 * s['REMOTE_H']['mean']:.1f} | {100 * s['OTHER']['mean']:.1f} | "
          f"{100 * s['REMOTE_C']['mean']:.1f} | "
          f"{100 * s['A_old']['mean']:.1f} | **{100 * s['A_new']['mean']:.1f}** | "
          f"{100 * s['B_old']['mean']:.1f} | {100 * s['B_new']['mean']:.1f} |")
    A('')

    # ---------- Table 3: per-atom intensity, atom-pooled
    A('## T3. Per-atom intensity ratio, atom-pooled (1.0 = the average atom)\n')
    A('| property | METAL | MBC | COORD | REMOTE_H | REMOTE_C |')
    A('|---|---|---|---|---|---|')
    for t in TARGETS:
        p = test['pooled_per_atom_ratio'][t]
        A(f"| {NAME[t]} | {p['METAL']:.2f} | {p['MBC']:.2f} | **{p['COORD']:.2f}** | "
          f"{p['REMOTE_H']:.2f} | {p['REMOTE_C']:.2f} |")
    A('')

    # ---------- Table 4: size-matched null
    A('## T4. Size-matched exchangeability null (per-molecule paired z)\n')
    A('| property | COORD mean | **COORD median** | REMOTE_H mean | '
      '**REMOTE_H median** | MBC mean | METAL mean | REMOTE_C mean |')
    A('|---|---|---|---|---|---|---|---|')
    for t in TARGETS:
        z = test['paired_z'][t]
        A(f"| {NAME[t]} | {z['COORD']['mean']:.2f} | "
          f"**{z['COORD']['median']:+.2f}** | {z['REMOTE_H']['mean']:.2f} | "
          f"**{z['REMOTE_H']['median']:+.2f}** | {z['MBC']['mean']:.2f} | "
          f"{z['METAL']['mean']:.2f} | {z['REMOTE_C']['mean']:.2f} |")
    A('')
    A(f"n = {test['paired_z']['homo']['COORD']['n']} complexes. z is the paired "
      'z against the exchangeability null E[share] = n_k/N.\n')
    A('The mean and the median disagree for uncoordinated donor-element atoms: '
      'their distribution is right-skewed, so the median is the honest summary '
      'of a *typical* complex. For HOMO the median is -0.2 (a typical complex '
      'gives uncoordinated donors no more weight than a random set of the same '
      'size) while the mean is +0.4, i.e. a minority of complexes carry a large '
      'remote contribution. Coordinated donors are unambiguous: median +2.7.\n')

    # ---------- Table 4b: distribution-free
    A('## T4b. Distribution-free form of T4 (test set)\n')
    A('Fraction of complexes in which the class exceeds its own size-matched '
      'expectation, and the mean excess in percentage points.\n')
    A('| property | COORD above | REMOTE_H above | MBC above | METAL above | '
      'REMOTE_C above | COORD excess | REMOTE_H excess | MBC excess |')
    A('|---|---|---|---|---|---|---|---|---|')
    for t in TARGETS:
        f = test['frac_above_null'][t]
        e = test['excess_pp'][t]
        A(f"| {NAME[t]} | **{100 * f['COORD']['mean']:.0f}%** | "
          f"{100 * f['REMOTE_H']['mean']:.0f}% | {100 * f['MBC']['mean']:.0f}% | "
          f"{100 * f['METAL']['mean']:.0f}% | {100 * f['REMOTE_C']['mean']:.0f}% | "
          f"{e['COORD']['mean']:+.1f} pp | {e['REMOTE_H']['mean']:+.1f} pp | "
          f"{e['MBC']['mean']:+.1f} pp |")
    A('')
    A('Under the null a class is above its expectation in 50% of complexes. '
      'COORD is at 75-85% for the five electronic/electrostatic properties; '
      'REMOTE_H is at 36-52%, i.e. at chance for a typical complex, while its '
      'mean paired z stays positive (T4) because a minority of complexes carry '
      'a strong remote contribution. Both statements are true and must be '
      'reported together: panel (a) of the published figure is dominated by '
      'coordinated donors, but not exclusively so.\n')

    # ---------- Table 5: the a -> b increase
    A('## T5. Is the Fig. 5a -> 5b increase explained by construction?\n')
    A('Observed increase vs the increase an exchangeable (size-matched) null '
      'predicts, which is exactly the growth of the set.\n')
    A('| property | A_old | B_old | observed increase | set grew by | ratio |')
    A('|---|---|---|---|---|---|')
    for t in TARGETS:
        a = test['share'][t]['A_old']['mean'] * 100
        b = test['share'][t]['B_old']['mean'] * 100
        sa = test['size_fraction']['A_old']['mean'] * 100
        sb = test['size_fraction']['B_old']['mean'] * 100
        A(f'| {NAME[t]} | {a:.1f}% | {b:.1f}% | {b - a:+.1f} pp | '
          f'{sb - sa:+.1f} pp | {(b - a) / (sb - sa):.2f}x |')
    A('')

    # ---------- Table 6: atom budget
    A('## T6. Atom budget (test-set sample)\n')
    A('| class | atoms | share of all atoms |')
    A('|---|---|---|')
    for c in ['METAL', 'MBC', 'COORD', 'REMOTE_H', 'OTHER', 'REMOTE_C']:
        A(f"| {c} | {test['meta']['atom_tally'][c]:,} | "
          f"{100 * test['meta']['atom_fraction'][c]:.2f}% |")
    A(f"\ntotal atoms: {test['meta']['n_atoms_total']:,}; "
      f"mean heavy atoms per complex: "
      f"{test['meta']['n_atoms_total'] / test['meta']['n_complexes_used']:.1f}")
    tally = test['meta']['atom_tally']
    n_tot = test['meta']['n_atoms_total']
    n_ao = tally['COORD'] + tally['REMOTE_H']
    A(f"\nAtom-pooled: A_old (published donor set) = {n_ao:,} atoms = "
      f"{100 * n_ao / n_tot:.2f}% of all atoms, of which REMOTE_H = "
      f"{tally['REMOTE_H']:,} = {100 * tally['REMOTE_H'] / n_tot:.2f} pp "
      f"**({100 * tally['REMOTE_H'] / n_ao:.1f}%) are NOT bonded to the metal**.")
    A(f"Per-complex mean (the quantity that enters Fig. 5): A_old = "
      f"{100 * test['size_fraction']['A_old']['mean']:.2f}%, REMOTE_H = "
      f"{100 * test['size_fraction']['REMOTE_H']['mean']:.2f}% "
      f"({100 * test['size_fraction']['REMOTE_H']['mean'] / test['size_fraction']['A_old']['mean']:.1f}%).\n")

    txt = '\n'.join(lines)
    out = os.path.join(RES, 'FIG5_result_tables_20261114.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)
    print(f'\n[save] {out}')


if __name__ == '__main__':
    main()
