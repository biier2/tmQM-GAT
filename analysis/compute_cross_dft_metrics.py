"""Compute supplementary cross-DFT metrics from external validation predictions.
Run on the server where external_validation.ipynb was executed (has graphs_gpu loaded).
If graphs not available, reads from saved npz file instead."""

import numpy as np, pandas as pd, json
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

targets = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']

# Option A: Load saved predictions (if available)
if os.path.exists('cross_dft_predictions.npz'):
    data = np.load('cross_dft_predictions.npz')
    y_true = data['y_true']
    smiles_preds = data['smiles_preds']
    print(f'Loaded from cross_dft_predictions.npz: {y_true.shape}')
else:
    # Option B: Read from external_validation_results.csv only
    print('No prediction file found. Reading from external_validation_results.csv...')
    print('Only Pearson r and MAE available in CSV.')
    if os.path.exists('external_validation_results.csv'):
        df = pd.read_csv('external_validation_results.csv')
        print(df.to_string(index=False))
    print('\nTo compute full metrics, save predictions to cross_dft_predictions.npz')
    print('with keys: y_true, smiles_preds')
    exit(1)

results = []
for i, prop in enumerate(targets):
    yt = y_true[:, i]
    yp_sm = smiles_preds[:, i]

    # Mask NaN
    mask = ~(np.isnan(yt) | np.isnan(yp_sm))
    yt_v, yp_v = yt[mask], yp_sm[mask]

    if len(yt_v) < 10:
        results.append({'Property': prop, 'error': 'too few valid samples'})
        continue

    # Regression
    A = np.vstack([yp_v, np.ones_like(yp_v)]).T
    slope, intercept = np.linalg.lstsq(A, yt_v, rcond=None)[0]

    # Metrics
    r_pearson, _ = pearsonr(yt_v, yp_v)
    rho, _ = spearmanr(yt_v, yp_v)
    r2 = r2_score(yt_v, yp_v)
    mae = mean_absolute_error(yt_v, yp_v)
    rmse = np.sqrt(mean_squared_error(yt_v, yp_v))

    # After linear recalibration
    yp_cal = yp_v * slope + intercept
    mae_cal = mean_absolute_error(yt_v, yp_cal)
    r2_cal = r2_score(yt_v, yp_cal)

    results.append({
        'Property': prop,
        'Pearson_r': float(r_pearson),
        'Spearman_rho': float(rho),
        'R2': float(r2),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'Slope': float(slope),
        'Intercept': float(intercept),
        'MAE_recal': float(mae_cal),
        'R2_recal': float(r2_cal),
    })

print(f'\n{"="*90}')
print('CROSS-DFT SUPPLEMENTARY METRICS (SMILES GAT on tmQMg, wB97xd/def2-SVP)')
print(f'{"="*90}')
print(f'{"Property":16s} {"Pearson r":>10s} {"Spearman rho":>12s} {"R2":>8s} {"MAE":>10s} {"RMSE":>10s} {"Slope":>8s} {"MAE_recal":>12s}')
print('-'*90)

for r in results:
    if 'error' in r:
        print(f'{r["Property"]:16s} ERROR: {r["error"]}')
    else:
        print(f'{r["Property"]:16s} {r["Pearson_r"]:10.4f} {r["Spearman_rho"]:12.4f} {r["R2"]:8.4f} {r["MAE"]:10.4f} {r["RMSE"]:10.4f} {r["Slope"]:8.4f} {r["MAE_recal"]:12.4f}')

with open('cross_dft_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved cross_dft_metrics.json')

# Summary
if not any('error' in r for r in results):
    mean_pearson = np.mean([r['Pearson_r'] for r in results])
    mean_spearman = np.mean([r['Spearman_rho'] for r in results])
    mean_r2 = np.mean([r['R2'] for r in results])
    print(f'\nMean Pearson r = {mean_pearson:.4f}')
    print(f'Mean Spearman rho = {mean_spearman:.4f}')
    print(f'Mean R2 = {mean_r2:.4f}')
