# -*- coding: utf-8 -*-
"""Remove the leftover smoke-test model (Fold0_Seed42) from ensemble_checkpoint_DimeNet.pt.

Background: on 2026-09-18, when the first official run started, the checkpoint
already contained a model left over from --smoke (128 molecules × 2 epochs); the
official run immediately did "Restored 1 models" + "SKIP Fold0_Seed42" and treated
it as a genuine ensemble member, dragging the 3-model ensemble mean R² down to
0.8317 (that bad model alone scored only 0.093, the other two were 0.898 / 0.905).

This script: back up the original checkpoint -> delete the (fold=0, seed=42) entry -> save it back.
Then re-run the official training; the script will SKIP the completed Seed123/456 and only train Seed42 (about 24 h).

Usage (in the chem_ml5 directory on the server):
    python reset_dimenet_checkpoint.py --dry-run   # just look first, change nothing
    python reset_dimenet_checkpoint.py             # execute after confirming
Then:
    nohup python server_dimenet_fixed.py > train_dimenet_seed42.log 2>&1 &
"""
import sys, os, shutil, argparse
sys.stdout.reconfigure(encoding='utf-8')
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='ensemble_checkpoint_DimeNet.pt')
ap.add_argument('--drop-fold', type=int, default=0)
ap.add_argument('--drop-seed', type=int, default=42)
ap.add_argument('--dry-run', action='store_true')
args = ap.parse_args()

if not os.path.exists(args.ckpt):
    sys.exit(f'Cannot find {args.ckpt} — if the official run has not started yet, just run server_dimenet_fixed.py directly')

ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
models, log = ck['models'], ck.get('log', [])

print(f'checkpoint: {args.ckpt}  ({os.path.getsize(args.ckpt)/1e6:.1f} MB)')
print(f'Currently {len(models)} models:')
for m, l in zip(models, log or [{}] * len(models)):
    print(f"  fold={m['fold']} seed={m['seed']}  training time={l.get('min','?')} min")

tgt = (args.drop_fold, args.drop_seed)
hit = [(m['fold'], m['seed']) for m in models if (m['fold'], m['seed']) == tgt]
if not hit:
    print(f'\nNo model ({args.drop_fold}, {args.drop_seed}) found, nothing to clean up.')
    sys.exit(0)

# Signature of a smoke-test model: extremely short training time (a few minutes)
l = next((x for x in log if (x['fold'], x['seed']) == tgt), {})
mins = l.get('min')
print(f'\nWill delete fold={args.drop_fold} seed={args.drop_seed} (training time {mins} min).')
if mins is not None and mins > 60:
    print('  ⚠️ Note: this model took > 60 min to train, so it does not look like a smoke-test model. Please confirm before executing.')

if args.dry_run:
    print('\n--dry-run: no changes made.')
    sys.exit(0)

bak = args.ckpt + '.before_reset'
if not os.path.exists(bak):
    shutil.copy2(args.ckpt, bak); print(f'Backup -> {bak}')

ck['models'] = [m for m in models if (m['fold'], m['seed']) != tgt]
ck['log']    = [x for x in log    if (x['fold'], x['seed']) != tgt]
torch.save(ck, args.ckpt)
print(f'Deleted. {len(ck["models"])} models remain now: '
      f'{[ (m["fold"], m["seed"]) for m in ck["models"] ]}')
print('\nNext step: nohup python server_dimenet_fixed.py > train_dimenet_seed42.log 2>&1 &')
print('Expected log: Restored 2 models -> SKIP Fold0_Seed123 -> SKIP Fold0_Seed456 '
      '-> Training Fold0_Seed42 (about 24 h)')
