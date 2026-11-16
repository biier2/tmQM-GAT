# -*- coding: utf-8 -*-
"""M7 diagnostics: quantifying the hardware sensitivity of the top-5 boundary
Problem: notebook (GPU, fp32) A1 = donor 20.5%, local CPU fp32 = 22.0% (the two RDKit/torch stacks are identical)
Diagnostics:
  1) rank5-rank6 score gap distribution (how many complexes have their boundary inside floating-point noise)
  2) fp32 vs fp64 ranking difference -> number of slot flips + f64 element distribution
  3) edge-order reversal perturbation (mimicking changes in GPU atomicAdd reduction order) -> flip count + element distribution
"""
import sys, json, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, torch
from collections import defaultdict
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from tqdm import tqdm

device = torch.device('cpu')
torch.set_num_threads(16)
torch.manual_seed(0)

# ===== Reuse the graph-building + model part of m7_carbon_donors_2024.py (up to before '# ===== 3.') =====
src = open('d:/ml/chem_ml5/m7_carbon_donors_2024.py', encoding='utf-8').read()
head = src.split('# ===== 3.')[0]
exec(head)

DONOR_CLS = {'N','O','S','P','Cl','Br','I'}
METAL_SYM = {'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ru','Rh','Pd','Ag','Cd','Pt','Au','Hg','Mo','W','Ir','Os','Re'}

# ===== Per complex: fp32 attention + gap statistics =====
gaps = []            # rank5 - rank6 (fp32, relative to rank5)
flip_f64 = 0         # number of f64 vs f32 slot flips
flip_rev = 0         # number of slot flips, reversed vs forward edge order
f64_counts = defaultdict(int)
rev_counts = defaultdict(int)
f32_counts = defaultdict(int)
boundary_donor_involved = 0   # at least one of rank5 / rank6 is a donor

for i in tqdm(range(len(xai_graphs)), desc='diag'):
    g = xai_graphs[i]; sym = xai_symbols[i]
    with torch.no_grad():
        _, attn_layers = model(g, return_attn=True)
    alpha = attn_layers[-1].mean(dim=1)          # [E]
    ei = g['edge_index']; dst = ei[1]

    # fp32 forward order (same as notebook/m7)
    na32 = torch.zeros(len(sym), dtype=torch.float32)
    na32.scatter_add_(0, dst, alpha)
    na32 = na32 / (na32.sum() + 1e-8)
    order32 = na32.argsort(descending=True)[:5].numpy()
    for idx in order32: f32_counts[sym[idx]] += 1

    # gap rank5-rank6 (treated as 0 when there are fewer than 6 atoms, necessarily flips)
    srt = torch.sort(na32, descending=True).values
    g5 = srt[4].item()
    g6 = srt[5].item() if len(srt) >= 6 else g5
    gaps.append(g5 - g6)
    full_order = np.argsort(na32.numpy())[::-1]
    r6 = sym[full_order[5]] if len(full_order) >= 6 else None
    if sym[order32[4]] in DONOR_CLS or (r6 is not None and r6 in DONOR_CLS):
        boundary_donor_involved += 1

    # fp64 reference ranking (edge_index kept as long, non-tensor keys passed through as is)
    g64 = {k: (v.double() if isinstance(v, torch.Tensor) and k != 'edge_index' else v)
           for k, v in g.items()}
    with torch.no_grad():
        model_d = model.double()
        _, attn_layers64 = model_d(g64, return_attn=True)
        model.float()  # restore
    alpha64 = attn_layers64[-1].mean(dim=1)
    na64 = torch.zeros(len(sym), dtype=torch.float64)
    na64.scatter_add_(0, dst, alpha64)
    na64 = na64 / (na64.sum() + 1e-8)
    order64 = na64.argsort(descending=True)[:5].numpy()
    for idx in order64: f64_counts[sym[idx]] += 1
    flip_f64 += len(set(order32) - set(order64))

    # Edge-order reversal (index dst in reverse order before accumulating -> reduction order changes, mimicking GPU atomic nondeterminism)
    rev = torch.arange(len(dst) - 1, -1, -1)
    na_rev = torch.zeros(len(sym), dtype=torch.float32)
    na_rev.scatter_add_(0, dst[rev], alpha[rev])
    na_rev = na_rev / (na_rev.sum() + 1e-8)
    order_rev = na_rev.argsort(descending=True)[:5].numpy()
    for idx in order_rev: rev_counts[sym[idx]] += 1
    flip_rev += len(set(order32) - set(order_rev))

total = len(xai_graphs)
T = total * 5
print(f'\nNumber of complexes: {total}')
gaps = np.array(gaps)
print(f'rank5-rank6 gap: median {np.median(gaps):.2e} | <1e-7: {(gaps<1e-7).mean()*100:.1f}% | <1e-6: {(gaps<1e-6).mean()*100:.1f}% | <1e-5: {(gaps<1e-5).mean()*100:.1f}% | <1e-4: {(gaps<1e-4).mean()*100:.1f}%')
print(f'Boundary involves a donor (rank5/rank6): {boundary_donor_involved} ({100*boundary_donor_involved/total:.1f}%)')
print(f'fp32 vs fp64: slot flips {flip_f64} ({100*flip_f64/T:.2f}% of slots)')
print(f'forward vs reverse order: slot flips {flip_rev} ({100*flip_rev/T:.2f}% of slots)')

def report(name, counts):
    d = sum(v for k, v in counts.items() if k in DONOR_CLS)
    m = sum(v for k, v in counts.items() if k in METAL_SYM)
    c = counts.get('C', 0)
    print(f'{name}: donor {100*d/T:.1f}% | C {100*c/T:.1f}% | metal {100*m/T:.1f}%')
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
    print(f'   top: {[(k, v) for k, v in top]}')

report('fp32 forward', f32_counts)
report('fp64      ', f64_counts)
report('fp32 reverse', rev_counts)

json.dump({
    'total': total,
    'gap_median': float(np.median(gaps)),
    'gap_lt_1e-7_pct': float((gaps < 1e-7).mean() * 100),
    'gap_lt_1e-6_pct': float((gaps < 1e-6).mean() * 100),
    'gap_lt_1e-5_pct': float((gaps < 1e-5).mean() * 100),
    'gap_lt_1e-4_pct': float((gaps < 1e-4).mean() * 100),
    'boundary_donor_pct': float(100 * boundary_donor_involved / total),
    'f64_flips': flip_f64, 'rev_flips': flip_rev,
    'f32': dict(f32_counts), 'f64': dict(f64_counts), 'rev': dict(rev_counts),
}, open('d:/ml/chem_ml5/m7_diag_sensitivity.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved -> m7_diag_sensitivity.json')
