# -*- coding: utf-8 -*-
"""Read-only recomputation (archived at the editing stage):
[A] 0.4 — conformal quantile "implementation (np.quantile linear interpolation, level=(1−α)(1+1/n))" vs
    "SI formula (direct order statistic k=ceil((n+1)(1−α)))" per property × 3 levels.
[B] id37 — gating reproduction: 11,556 evaluation subset, independent per property, hw@90 upper-quartile rejection, count
    and MAE comparison (SI tables S14/S15: 8,667 (75.0%), 0.0078/1.105, 0.0061/0.879,
    pre-gating 0.0092/1.394), and export of the evaluation indices and per-property retained masks.
Data and definitions = _audit_conformal.py (RandomState(42), ddof=0, σ≥1e-6, unclipped mean).
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
# Input defaults to the archived prediction copy in the repository; override
# with RECON_PREDS / RECON_OUT if needed.
IN_NPZ = os.environ.get('RECON_PREDS',
                        os.path.join(REPO, 'preds', 'test_preds_A_8d6d.npz'))
OUT = os.environ.get('RECON_OUT', os.path.join(HERE, '_out'))
os.makedirs(OUT, exist_ok=True)
P8 = ["HL-Gap", "HOMO", "LUMO", "Dipole", "MetalCharge", "E_elec", "E_disp", "Polarizability"]
PRINTED_HW = {'HL-Gap': 0.020, 'HOMO': 0.019, 'LUMO': 0.018, 'Dipole': 3.42,
              'MetalCharge': 0.18, 'E_elec': 406.0, 'E_disp': 0.028, 'Polarizability': 56.9}
PRINTED_PRE = {'HL-Gap': 0.0092, 'Dipole': 1.394}
PRINTED_POST = {'HL-Gap': 0.0078, 'Dipole': 1.105}
PRINTED_Q1 = {'HL-Gap': 0.0061, 'Dipole': 0.879}

print('input:', IN_NPZ)
z = np.load(IN_NPZ, allow_pickle=True)
print('npz keys:', z.files)
pm = np.asarray(z['preds'], dtype=np.float64)   # (9, N, 8)
y_true = np.asarray(z['y_true'], dtype=np.float64)
n = len(y_true)
if n != 13556:
    raise SystemExit(f'test rows = {n}, expected 13,556 (wrong prediction archive?)')
y_mean = pm.mean(axis=0)
sigma = np.maximum(pm.std(axis=0), 1e-6)
rng = np.random.RandomState(42)
cal_idx = rng.choice(n, size=2000, replace=False)
mask = np.zeros(n, bool); mask[cal_idx] = True
ev = ~mask
print('N=%d  calibration=%d  evaluation=%d' % (n, mask.sum(), ev.sum()))
ev_idx = np.where(ev)[0]
n_cal = int(mask.sum())

# ---------- [A] Quantile-definition comparison ----------
print('\n===== [A] 0.4 quantile comparison: interp vs order-stat =====')
print('%-14s %5s | %12s %12s %10s | %9s %9s | %9s %9s' %
      ('property', 'level', 'q_interp', 'q_order', 'dq', 'cov_int', 'cov_ord', 'hw_int', 'hw_ord'))
qcmp = {}
for j, p in enumerate(P8):
    for tgt in (0.80, 0.90, 0.95):
        a = 1 - tgt
        s_cal = np.abs(y_true[mask, j] - y_mean[mask, j]) / sigma[mask, j]
        q_i = float(np.quantile(s_cal, min(1.0, (1 - a) * (1 + 1.0 / n_cal)),
                                method='linear'))
        k = int(np.ceil((n_cal + 1) * (1 - a)))
        q_o = float(np.sort(s_cal)[k - 1])
        s_ev = np.abs(y_true[ev, j] - y_mean[ev, j]) / sigma[ev, j]
        cov_i = float((s_ev <= q_i).mean())
        cov_o = float((s_ev <= q_o).mean())
        hw_i = float(2 * q_i * sigma[ev, j].mean())
        hw_o = float(2 * q_o * sigma[ev, j].mean())
        qcmp['%s@%.2f' % (p, tgt)] = dict(q_interp=q_i, q_order=q_o, cov_interp=cov_i,
                                          cov_order=cov_o, hw_interp=hw_i, hw_order=hw_o)
        print('%-14s %.2f  | %12.6f %12.6f %10.2e | %9.5f %9.5f | %9.5f %9.5f' %
              (p, tgt, q_i, q_o, q_o - q_i, cov_i, cov_o, hw_i, hw_o))
dqs = [abs(v['q_order'] - v['q_interp']) / max(v['q_interp'], 1e-12) for v in qcmp.values()]
dcovs = [abs(v['cov_order'] - v['cov_interp']) for v in qcmp.values()]
print('max relative difference |dq|/q = %.3e ; max |Δcov| = %.5f (24 combinations)' % (max(dqs), max(dcovs)))
rngs = {t: (min(v['cov_interp'] for kk, v in qcmp.items() if kk.endswith('%.2f' % t)),
            max(v['cov_interp'] for kk, v in qcmp.items() if kk.endswith('%.2f' % t)))
        for t in (0.80, 0.90, 0.95)}
print('interp coverage ranges: 0.80→%.4f–%.4f | 0.90→%.4f–%.4f | 0.95→%.4f–%.4f  (SI: 0.784–0.818 / 0.880–0.908 / 0.937–0.956)'
      % (rngs[0.80] + rngs[0.90] + rngs[0.95]))

# ---------- [B] Gating reproduction ----------
print('\n===== [B] id37 gating reproduction (90% half-width, independent per property) =====')
gate = {}
keep_masks = {}
for j, p in enumerate(P8):
    a = 1 - 0.90
    s_cal = np.abs(y_true[mask, j] - y_mean[mask, j]) / sigma[mask, j]
    q90 = float(np.quantile(s_cal, min(1.0, (1 - a) * (1 + 1.0 / n_cal)),
                            method='linear'))
    hw = q90 * sigma[ev, j]                       # per-complex half-width (= q*σ; the mean in table 14 = q*mean σ)
    mae_pre = float(np.abs(y_true[ev, j] - y_mean[ev, j]).mean())
    thr75 = float(np.percentile(hw, 75))
    thr25 = float(np.percentile(hw, 25))
    keep75 = hw <= thr75
    keep25 = hw <= thr25
    lt75 = int((hw < thr75).sum()); le75 = int(keep75.sum()); eq75 = int((hw == thr75).sum())
    mae_post = float(np.abs(y_true[ev, j][keep75] - y_mean[ev, j][keep75]).mean())
    mae_q1 = float(np.abs(y_true[ev, j][keep25] - y_mean[ev, j][keep25]).mean())
    gate[p] = dict(q90=q90, hw_mean=float(hw.mean()), mae_pre=mae_pre, thr75=thr75,
                   retained_le=le75, retained_lt=lt75, ties_at_thr=eq75, mae_post=mae_post,
                   retained_q1=int(keep25.sum()), mae_q1=mae_q1)
    keep_masks[p] = keep75
    print('%-14s q90=%.5f hw_mean=%.5f(printed %.3f) | pre %.4f | >75%% retained ≤%d  < %d (ties at thr %d) | post %.4f | lowest quartile %d, MAE %.4f' %
          (p, q90, hw.mean(), PRINTED_HW[p], mae_pre, le75, lt75, eq75, mae_post, keep25.sum(), mae_q1))
print()
print('Against SI table S15: pre-gating (11,556) HL-Gap %.4f (printed 0.0092), Dipole %.4f (printed 1.394); rejecting the upper quartile retains %d (printed 8,667) → HL-Gap %.4f (printed 0.0078), Dipole %.4f (printed 1.105); lowest quartile %d cases (printed 2,889) → HL-Gap %.4f (printed 0.0061), Dipole %.4f (printed 0.879)'
      % (gate['HL-Gap']['mae_pre'], gate['Dipole']['mae_pre'],
         gate['HL-Gap']['retained_le'],
         gate['HL-Gap']['mae_post'], gate['Dipole']['mae_post'],
         gate['HL-Gap']['retained_q1'], gate['HL-Gap']['mae_q1'], gate['Dipole']['mae_q1']))

# ---------- Archived outputs ----------
with open(os.path.join(OUT, 'conformal_quantile_compare.json'), 'w', encoding='utf-8') as f:
    json.dump(qcmp, f, indent=1, ensure_ascii=False)
with open(os.path.join(OUT, 'gating_recon.json'), 'w', encoding='utf-8') as f:
    json.dump({'N_eval': int(ev.sum()), 'gate': gate}, f, indent=1, ensure_ascii=False)
np.savez(os.path.join(OUT, 'gating_indices.npz'),
         ev_idx=ev_idx, cal_idx=cal_idx,
         **{('keep_' + p.replace(' ', '')): keep_masks[p] for p in P8})
print('\n-> conformal_quantile_compare.json / gating_recon.json / gating_indices.npz')
