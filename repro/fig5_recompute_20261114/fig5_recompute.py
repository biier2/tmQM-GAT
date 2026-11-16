# -*- coding: utf-8 -*-
"""Fig. 5 recomputation on the held-out 13,556-complex test set (R1-2 + R2-4).

Protocol follows the published Fig. 5 exactly (main text P108-P113 / SI S5.2):
  * nine-member variant-A ensemble, member-mean signed node-feature occlusion
  * attributions inverse-transformed to original property units before |.| is taken
  * per-property "normalized absolute attribution fraction"

What is new here, and why:
  1. SAMPLE.  The published 2,000 complexes were drawn from the full 90,377-complex
     retained domain ("not restricted to the test set", SI P122); ~85% of that pool
     is training/CV data (R1-2).  This script re-runs the identical protocol on
     2,000 complexes drawn from the fixed 13,556-complex test set.
  2. ATOM CLASSES.  The published donor set is the element set {N,O,S,P,Cl,Br,I},
     which counts heteroatoms that are NOT bonded to the metal (R2-4).  Here every
     atom is typed from the graph itself, so that COORD+REMOTE_H reproduces that
     element set EXACTLY (F, B, Si, ... are NOT donors and go to OTHER):
        METAL     transition metal
        MBC       carbon bonded to the metal (metal-bonded carbon)
        COORD     donor element bonded to the metal (true donor)
        REMOTE_H  donor element NOT bonded to the metal (same element, long range)
        OTHER     heteroatom outside the published donor set (F, B, Si, Se, ...)
        REMOTE_C  carbon not bonded to the metal
     and the published sets are reconstructed as COORD+REMOTE_H (panel a) and
     COORD+REMOTE_H+MBC (panel b) so the correction can be quantified.
  3. PER-ATOM INTENSITY.  mean |dPred| of a class divided by the molecule-wide
     mean |dPred| -> enrichment per atom (1.0 = an average atom).  Share alone
     confounds class size with class importance.
  4. SIZE-MATCHED NULL.  Exchangeability null within each molecule: E[share]=n_k/N,
     Var from sampling without replacement; per-molecule paired z.  A numerical
     permutation check is run on the first --perm-check molecules.

Usage:
    python fig5_recompute.py --sample test        --models ensemble
    python fig5_recompute.py --sample fulldomain  --models ensemble   # reproduction
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('TMQM_DATA_DIR', r'D:\ml\chem_ml5')

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge',
           'E_elec', 'E_disp', 'polarizability']
HIDDEN, N_LAYERS, GAT_HEADS = 384, 5, 12

METALS = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
          'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
          'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'}
Z2SYM = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P',
         16: 'S', 17: 'Cl', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co',
         28: 'Ni', 29: 'Cu', 30: 'Zn', 35: 'Br', 40: 'Zr', 41: 'Nb', 42: 'Mo',
         44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 53: 'I', 72: 'Hf',
         73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
         80: 'Hg'}
# the published Fig. 5 donor set, verbatim from SI S5.2 (P126)
DONOR_ELEMS = {'N', 'O', 'S', 'P', 'Cl', 'Br', 'I'}
CLASSES = ['METAL', 'MBC', 'COORD', 'REMOTE_H', 'OTHER', 'REMOTE_C']
# legend: published sets vs corrected sets
SETS = {
    'A_old': ['COORD', 'REMOTE_H'],            # published panel (a) == the element set
    'A_new': ['COORD'],                        # coordinated donors only
    'B_old': ['COORD', 'REMOTE_H', 'MBC'],     # published panel (b)
    'B_new': ['COORD', 'MBC'],                 # coordinated donors + metal-bonded C
    'REMOTE_H': ['REMOTE_H'],
    'MBC': ['MBC'],
    'METAL': ['METAL'],
    'REMOTE_C': ['REMOTE_C'],
    'OTHER': ['OTHER'],
}


# ---------------------------------------------------------------- model
def segment_softmax(scores, dst, N):
    s32 = scores.float()
    max_v = torch.full((N, s32.size(1)), float('-inf'), dtype=torch.float32)
    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, s32.size(1)), s32,
                          reduce='amax', include_self=False)
    exp_s = torch.exp(s32 - max_v[dst])
    sum_v = torch.zeros(N, s32.size(1), dtype=torch.float32)
    sum_v.scatter_add_(0, dst.unsqueeze(-1).expand(-1, s32.size(1)), exp_s)
    return (exp_s / (sum_v[dst] + 1e-8)).to(scores.dtype)


class GATConvXAI(nn.Module):
    def __init__(self, in_dim, out_dim_per_head, edge_dim=0, heads=4):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim_per_head
        total_dim = out_dim_per_head * heads
        self.node_lin = nn.Linear(in_dim, total_dim, bias=False)
        self.attn_src = nn.Linear(total_dim, heads, bias=False)
        self.attn_dst = nn.Linear(total_dim, heads, bias=False)
        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim > 0 else None
        self.act = nn.ReLU()

    def forward(self, x, ei, ea=None):
        src, dst = ei[0], ei[1]
        N = x.size(0)
        h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0:
            attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, N)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1)
        msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(N, self.out_dim * self.heads, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out), alpha


def scatter_mean_per_graph(h, batch, n):
    out = torch.zeros(n, h.size(1), dtype=h.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
    cnt = torch.zeros(n, dtype=h.dtype)
    cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)


def scatter_max_per_graph(h, batch, n):
    out = torch.full((n, h.size(1)), float('-inf'), dtype=h.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce='amax', include_self=False)
    return out


class XYZ_GNN_XAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, HIDDEN)
        self.convs = nn.ModuleList([GATConvXAI(HIDDEN, HIDDEN // GAT_HEADS, 6, heads=GAT_HEADS)
                                    for _ in range(N_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
        self.drop = nn.Dropout(0.10)
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN * 3, HIDDEN), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HIDDEN // 2, 8))

    def forward(self, g, batch=None, return_attn=False):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        all_attn = []
        for conv, norm in zip(self.convs, self.norms):
            h_new, alpha = conv(h, ei, ea)
            h = norm(h + h_new)
            h = self.drop(h)
            if return_attn:
                all_attn.append(alpha)
        if batch is None:
            mm = (x[:, 7] > 0.5)
            mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), dtype=h.dtype)
            out = self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        else:
            n_graphs = int(batch.max().item()) + 1
            mean_h = scatter_mean_per_graph(h, batch, n_graphs)
            max_h = scatter_max_per_graph(h, batch, n_graphs)
            mm = (x[:, 7] > 0.5)
            mf = (scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any()
                  else torch.zeros(n_graphs, h.size(1), dtype=h.dtype))
            out = self.readout(torch.cat([mean_h, max_h, mf], dim=1))
        if return_attn:
            return out, all_attn
        return out


# ---------------------------------------------------------------- data
def load_models(which):
    """Return [(module, y_std_np or None, tag), ...]."""
    out = []
    if which == 'ensemble':
        ck = torch.load(os.path.join(DATA_DIR, 'ensemble_checkpoint_A_8d6d.pt'),
                        map_location='cpu', weights_only=False)
        for mm in ck['models']:
            m = XYZ_GNN_XAI()
            m.load_state_dict(mm['state'], strict=True)
            m.eval()
            out.append((m, mm['y_std'].numpy().astype(np.float64),
                        f'fold{mm["fold"]}_seed{mm["seed"]}'))
    else:
        sd = torch.load(os.path.join(DATA_DIR, 'tmqm_gat_best.pt'), map_location='cpu')
        m = XYZ_GNN_XAI()
        m.load_state_dict(sd, strict=True)
        m.eval()
        out.append((m, None, 'tmqm_gat_best.pt'))
    return out


def build_old_sample(D, n, seed=123):
    """Reproduce the published full-domain draw: permute the raw CSV with seed 42,
    keep the first 20,000 that parse, then RandomState(123).choice of 2,000."""
    ref_smi = None
    csv_path = os.path.join(DATA_DIR, 'data_tmQM', 'local_vB', 'tmQM_y.csv')
    import pandas as pd
    df = pd.read_csv(csv_path, sep=';')
    smis = df['SMILES'].tolist()
    rng = np.random.RandomState(42)
    built = []
    for csv_i in rng.permutation(len(df)):
        if len(built) >= 20000:
            break
        s = smis[csv_i]
        if not isinstance(s, str) or s == '':
            continue
        if Chem.MolFromSmiles(s) is None:
            continue
        built.append(int(csv_i))
    built = np.asarray(built)
    pick = np.random.RandomState(seed).choice(len(built), n, replace=False)
    old_rows = built[pick]
    csv_idx = np.asarray(D['csv_idx'])
    pos = {int(c): i for i, c in enumerate(csv_idx)}
    hits = [pos[c] for c in old_rows if int(c) in pos]
    missing = int(len(old_rows) - len(hits))
    print(f'[old sample] reproduced {len(old_rows)} CSV rows (seed 42/123); '
          f'{len(hits)} present in the 90,377 retained set, {missing} not')
    del ref_smi
    return np.asarray(hits)


def sample_indices(D, kind, n, seed):
    if kind == 'test':
        te = np.asarray(D['te_idx'])
        pick = np.sort(np.random.RandomState(seed).choice(len(te), n, replace=False))
        return te[pick], te
    hits = build_old_sample(D, n)
    return hits, np.asarray(D['te_idx'])


def classify_atoms(z, ei, ea, rule='edge'):
    """rule='edge' identifies coordination topologically (an edge to a metal
    node); rule='flag' uses edge feature 5.  The stored flag is 0 for every
    metal-Br, metal-I and metal-Se edge in new_domain_iso.pt even though those
    edges exist, so 'edge' is the chemically correct choice."""
    N = len(z)
    # full periodic table, so that Sc/Ti/Y/Tc are not silently lost as "unknown"
    pt = Chem.GetPeriodicTable()
    sym = [pt.GetElementSymbol(int(v)) for v in z]
    is_metal = np.array([s in METALS for s in sym])
    src, dst = ei[0].numpy(), ei[1].numpy()
    if rule == 'edge':
        ml = is_metal[src] ^ is_metal[dst]
    else:
        ml = ea[:, 5].numpy() > 0.5
    bonded = np.zeros(N, dtype=bool)
    bonded[src[ml & ~is_metal[src]]] = True
    bonded[dst[ml & ~is_metal[dst]]] = True
    cls = np.empty(N, dtype=object)
    for i in range(N):
        if is_metal[i]:
            cls[i] = 'METAL'
        elif z[i] == 6:
            cls[i] = 'MBC' if bonded[i] else 'REMOTE_C'
        elif sym[i] in DONOR_ELEMS:
            cls[i] = 'COORD' if bonded[i] else 'REMOTE_H'
        else:
            cls[i] = 'OTHER'          # F, B, Si, Se, ... - never in the published set
    return sym, cls


# ---------------------------------------------------------------- occlusion
def build_occlusion_batch(x, ei, ea):
    Na = x.size(0)
    xb = x.repeat(Na, 1).clone()
    xb[torch.arange(Na) * Na + torch.arange(Na)] = 0.0
    E = ei.size(1)
    off = (torch.arange(Na) * Na).repeat_interleave(E).view(1, -1)
    return ({'x': xb, 'edge_index': ei.repeat(1, Na) + off, 'edge_attr': ea.repeat(Na, 1)},
            torch.arange(Na).repeat_interleave(Na))


def occlusion_attribution(model, x, ei, ea):
    """Return (base (8,), occluded (Na,8)) in the model's output space."""
    with torch.no_grad():
        base = model({'x': x, 'edge_index': ei, 'edge_attr': ea})[0]
        gb, batch = build_occlusion_batch(x, ei, ea)
        occ = model(gb, batch=batch)
    return base.numpy(), occ.numpy()


# ---------------------------------------------------------------- null model
def analytic_null(w, mask):
    N = len(w)
    n = int(mask.sum())
    if n == 0 or n == N:
        return np.nan, np.nan, np.nan
    tot = w.sum()
    if tot <= 0 or N < 2:
        return np.nan, np.nan, np.nan
    obs = w[mask].sum() / tot
    mu = n / N
    var_sum = n * (N - n) / (N - 1) * w.var()
    sd = np.sqrt(var_sum) / tot
    return obs, mu, ((obs - mu) / sd if sd > 0 else np.nan)


def permutation_null(w, mask, R, rng):
    N = len(w)
    tot = w.sum()
    if tot <= 0 or mask.sum() == 0:
        return np.nan, np.nan
    perms = np.argsort(rng.random((R, N)), axis=1)
    sh = w[perms][:, mask].sum(axis=1) / tot
    return float(sh.mean()), float(sh.std())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='test', choices=['test', 'fulldomain'])
    ap.add_argument('--models', default='ensemble', choices=['single', 'ensemble'])
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261114)
    ap.add_argument('--perm-check', type=int, default=100)
    ap.add_argument('--avg', default='pred', choices=['pred', 'abs'],
                    help='pred = published protocol (average predictions, then subtract); '
                         'abs = average of per-member |delta|')
    ap.add_argument('--out', default=None)
    ap.add_argument('--bond-rule', default='edge', choices=['edge', 'flag'],
                    help='how to decide that an atom is bonded to the metal')
    ap.add_argument('--dump-atoms', action='store_true',
                    help='store per-atom symbols/classes/attributions for every '
                         'complex (needed for the within-element bonded vs '
                         'non-bonded comparison, which is free of the '
                         'heteroatom-versus-carbon confound)')
    args = ap.parse_args()
    out_path = args.out or os.path.join(HERE, f'fig5_{args.sample}_{args.models}_{args.avg}.json')

    print('[data] loading new_domain_iso.pt ...', flush=True)
    D = torch.load(os.path.join(DATA_DIR, 'new_domain_iso.pt'), map_location='cpu',
                   weights_only=False)
    print(f'[data] N={D["y_all"].size(0):,}  test={len(D["te_idx"]):,}')

    sel, te = sample_indices(D, args.sample, args.n, args.seed)
    print(f'[sample] {args.sample}: {len(sel):,} complexes '
          f'(test fraction in this sample: {100 * np.isin(sel, te).mean():.1f}%)')

    models = load_models(args.models)
    print(f'[model] {args.models}: {len(models)} model(s) -> {[m[2] for m in models]}')

    keys = list(SETS.keys())
    occ = {t: {k: [] for k in keys} for t in TARGETS}
    inten = {t: {k: [] for k in keys} for t in TARGETS}
    zz = {t: {k: [] for k in keys} for t in TARGETS}
    excess = {t: {k: [] for k in keys} for t in TARGETS}   # obs - null, in pp
    pos = {t: {k: [] for k in keys} for t in TARGETS}      # 1 if obs > null
    size = {k: [] for k in keys}
    atom_tally = {c: 0 for c in CLASSES}
    # atom-pooled intensities: sum of |dPred| over every atom of the set, per property
    pooled_sum = {(t, k): 0.0 for t in TARGETS for k in keys}
    pooled_cnt = {k: 0 for k in keys}
    pooled_natoms = 0
    total_sum = np.zeros(len(TARGETS))
    n_atoms_all = 0
    perm_store = []
    examples = []
    atoms_dump = [] if args.dump_atoms else None

    t0 = time.time()
    for k, gi in enumerate(sel):
        gi = int(gi)
        x = torch.as_tensor(D['node8s'][gi], dtype=torch.float32)
        ei = D['ei_ls'][gi].long()
        ea = D['e6_ls'][gi].float()
        z = x[:, 0].numpy().astype(int)
        sym, cls = classify_atoms(z, ei, ea, args.bond_rule)
        Na = len(z)
        n_atoms_all += Na
        for c in CLASSES:
            atom_tally[c] += int((cls == c).sum())

        # published protocol (SI P122): inverse-transform each member to original
        # units, ensemble-average, then take the signed baseline-minus-masked delta.
        base_acc = np.zeros(len(TARGETS))
        occ_acc = np.zeros((Na, len(TARGETS)))
        abs_acc = np.zeros((Na, len(TARGETS)))
        for (m, ys, tag) in models:
            b, o = occlusion_attribution(m, x, ei, ea)
            if ys is not None:
                b = b * ys                    # y_mean cancels in baseline-minus-masked
                o = o * ys[None, :]
            base_acc += b
            occ_acc += o
            abs_acc += np.abs(b[None, :] - o)
        if args.avg == 'pred':
            dabs = np.abs(base_acc[None, :] - occ_acc) / len(models)
        else:
            dabs = abs_acc / len(models)
        w0 = dabs.sum(axis=1)
        mean_all = w0.mean() + 1e-30
        if atoms_dump is not None:
            atoms_dump.append({'idx': gi, 'sym': np.array(sym), 'cls': np.array(cls),
                               'dabs': dabs.astype(np.float32)})

        for kk in keys:
            mk = np.isin(cls, SETS[kk])
            size[kk].append(mk.sum() / Na)
            pooled_cnt[kk] += int(mk.sum())
        pooled_natoms += Na
        for j, t in enumerate(TARGETS):
            w = dabs[:, j]
            total_sum[j] += float(w.sum())
            for kk in keys:
                mk = np.isin(cls, SETS[kk])
                pooled_sum[(t, kk)] += float(w[mk].sum())
                obs, mu, zval = analytic_null(w, mk)
                if not np.isnan(obs):
                    occ[t][kk].append(obs)
                    zz[t][kk].append(zval)
                    excess[t][kk].append((obs - mu) * 100)
                    pos[t][kk].append(1.0 if obs > mu else 0.0)
                if mk.sum() > 0:
                    inten[t][kk].append(float(w[mk].mean() / (w.mean() + 1e-30)))

        if args.perm_check and k < args.perm_check:
            prng = np.random.RandomState(7000 + k)
            rec = {}
            for j, t in enumerate(TARGETS):
                for kk in keys:
                    mk = np.isin(cls, SETS[kk])
                    if 0 < mk.sum() < Na:
                        rec[f'{t}|{kk}'] = permutation_null(dabs[:, j], mk, 100, prng)
            perm_store.append(rec)

        if k < 60:
            examples.append({'idx': gi, 'csd': D['csd_codes'][gi], 'sym': sym,
                             'cls': cls.tolist(), 'dabs': dabs.tolist()})

        if (k + 1) % 200 == 0:
            el = time.time() - t0
            print(f'  {k + 1}/{len(sel)}  {el:.0f}s ({el / (k + 1):.3f}s/mol)', flush=True)

    print(f'[done] {len(sel)} molecules in {time.time() - t0:.1f}s')

    def summ(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return None
        return dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std()),
                    median=float(np.median(v)))

    out = {
        'meta': {
            'sample': args.sample,
            'n_complexes_requested': args.n,
            'n_complexes_used': int(len(sel)),
            'n_complexes_in_test_set_total': int(len(te)),
            'test_fraction_of_sample': float(np.isin(sel, te).mean()),
            'seed': args.seed,
            'model': args.models,
            'n_models': len(models),
            'avg_order': args.avg,
            'targets': TARGETS,
            'sets': {k: v for k, v in SETS.items()},
            'n_atoms_total': int(n_atoms_all),
            'atom_tally': atom_tally,
            'atom_fraction': {c: atom_tally[c] / max(n_atoms_all, 1) for c in CLASSES},
            'units': 'attributions inverse-transformed to original property units per member',
        },
        'share': {t: {k: summ(occ[t][k]) for k in keys} for t in TARGETS},
        'per_atom_ratio': {t: {k: summ(inten[t][k]) for k in keys} for t in TARGETS},
        'paired_z': {t: {k: summ(zz[t][k]) for k in keys} for t in TARGETS},
        'excess_pp': {t: {k: summ(excess[t][k]) for k in keys} for t in TARGETS},
        'frac_above_null': {t: {k: summ(pos[t][k]) for k in keys} for t in TARGETS},
        'size_fraction': {k: summ(size[k]) for k in keys},
    }
    # atom-pooled: divide by atom counts, then normalise by the all-atom pooled mean
    pooled = {}
    for j, t in enumerate(TARGETS):
        grand = total_sum[j] / max(pooled_natoms, 1)
        pooled[t] = {k: float((pooled_sum[(t, k)] / max(pooled_cnt[k], 1)) / (grand + 1e-30))
                     for k in keys}
    out['pooled_per_atom_ratio'] = pooled
    out['pooled_atom_counts'] = {k: int(pooled_cnt[k]) for k in keys}
    if perm_store:
        agg = {}
        for rec in perm_store:
            for k2, (mu, sd) in rec.items():
                agg.setdefault(k2, []).append(mu)
        out['permutation_check'] = {k2: dict(n=len(v), mean_null=float(np.mean(v)))
                                    for k2, v in agg.items()}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f'[save] {out_path}')

    np.savez(os.path.join(HERE, f'examples_{args.sample}_{args.models}.npz'),
             rows=np.array(examples, dtype=object), allow_pickle=True)
    if atoms_dump is not None:
        ap_out = os.path.join(HERE, f'atoms_{args.sample}_{args.models}.npz')
        np.savez(ap_out, rows=np.array(atoms_dump, dtype=object), allow_pickle=True)
        print(f'[save] {ap_out}')

    # ---------------- console summary ----------------
    print('\n=== atom budget ===')
    for c in CLASSES:
        print(f'  {c:9s} {atom_tally[c]:7d} atoms  {100 * atom_tally[c] / max(n_atoms_all, 1):5.2f}%')
    print('\n=== attribution share of total |dPred| (%) ===')
    hdr = '  ' + 'property'.ljust(14) + ''.join(c.rjust(9) for c in keys)
    print(hdr)
    for t in TARGETS:
        line = '  ' + t.ljust(14)
        for c in keys:
            s = out['share'][t][c]
            line += (f'{100 * s["mean"]:8.2f}%' if s else '        -')
        print(line)
    print('\n=== per-atom intensity ratio (1.0 = average atom) ===')
    print(hdr)
    for t in TARGETS:
        line = '  ' + t.ljust(14)
        for c in keys:
            s = out['per_atom_ratio'][t][c]
            line += (f'{s["mean"]:9.2f}' if s else '        -')
        print(line)
    print('\n=== size-matched paired z ===')
    print(hdr)
    for t in TARGETS:
        line = '  ' + t.ljust(14)
        for c in keys:
            s = out['paired_z'][t][c]
            line += (f'{s["mean"]:9.2f}' if s else '        -')
        print(line)
    print('\n=== size fraction (%) ===')
    for c in keys:
        s = out['size_fraction'][c]
        print(f'  {c:9s} {100 * s["mean"]:6.2f}%')


if __name__ == '__main__':
    main()
