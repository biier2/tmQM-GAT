# -*- coding: utf-8 -*-
"""清掉 ensemble_checkpoint_DimeNet.pt 里的冒烟废模型（Fold0_Seed42）。

背景：2026-09-18 首次正式跑时，checkpoint 里已有一个 --smoke 留下的模型
（128 分子 × 2 epoch），正式跑一启动就 "Restored 1 models" + "SKIP Fold0_Seed42"，
把它当成正式集成成员，3 模型集成 mean R² 被拖到 0.8317（该废模型单体只有 0.093，
另两个是 0.898 / 0.905）。

本脚本：备份原 checkpoint -> 删除 (fold=0, seed=42) 的条目 -> 存回。
之后重跑正式训练，脚本会 SKIP 已完成的 Seed123/456、只补训 Seed42（约 24 h）。

用法（在服务器 chem_ml5 目录）：
    python reset_dimenet_checkpoint.py --dry-run   # 先只看，不动
    python reset_dimenet_checkpoint.py             # 确认后执行
然后：
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
    sys.exit(f'找不到 {args.ckpt} —— 若正式跑还没开始，直接跑 server_dimenet_fixed.py 即可')

ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
models, log = ck['models'], ck.get('log', [])

print(f'checkpoint: {args.ckpt}  （{os.path.getsize(args.ckpt)/1e6:.1f} MB）')
print(f'当前 {len(models)} 个模型：')
for m, l in zip(models, log or [{}] * len(models)):
    print(f"  fold={m['fold']} seed={m['seed']}  训练耗时={l.get('min','?')} min")

tgt = (args.drop_fold, args.drop_seed)
hit = [(m['fold'], m['seed']) for m in models if (m['fold'], m['seed']) == tgt]
if not hit:
    print(f'\n没有 ({args.drop_fold}, {args.drop_seed}) 这个模型，无需清理。')
    sys.exit(0)

# 冒烟模型的特征：训练耗时极短（几分钟）
l = next((x for x in log if (x['fold'], x['seed']) == tgt), {})
mins = l.get('min')
print(f'\n将删除 fold={args.drop_fold} seed={args.drop_seed}（训练耗时 {mins} min）。')
if mins is not None and mins > 60:
    print('  ⚠️ 注意：该模型训练耗时 > 60 min，看起来不像冒烟模型。请先确认再执行。')

if args.dry_run:
    print('\n--dry-run：未做任何修改。')
    sys.exit(0)

bak = args.ckpt + '.before_reset'
if not os.path.exists(bak):
    shutil.copy2(args.ckpt, bak); print(f'备份 -> {bak}')

ck['models'] = [m for m in models if (m['fold'], m['seed']) != tgt]
ck['log']    = [x for x in log    if (x['fold'], x['seed']) != tgt]
torch.save(ck, args.ckpt)
print(f'已删除。现在剩 {len(ck["models"])} 个模型：'
      f'{[ (m["fold"], m["seed"]) for m in ck["models"] ]}')
print('\n下一步：nohup python server_dimenet_fixed.py > train_dimenet_seed42.log 2>&1 &')
print('预期日志：Restored 2 models -> SKIP Fold0_Seed123 -> SKIP Fold0_Seed456 '
      '-> Training Fold0_Seed42（约 24 h）')
