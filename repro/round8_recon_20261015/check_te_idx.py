# -*- coding: utf-8 -*-
"""只读核对：13,556 vs 13,557 测试集口径（代理报告项3/5 的警示项）。
比对 four sources：new_domain_iso.pt / ensemble_split_new.npz / test_preds_A_8d6d.npz / test_preds_DimeNet.npz。
另列 server_package 目录，找 README_dimenet_fixed.md。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
R = r'D:\ml\chem_ml5'

d = torch.load(os.path.join(R, 'new_domain_iso.pt'), map_location='cpu', weights_only=False)
te_iso = np.asarray(d['te_idx'])
print('new_domain_iso.pt : N =', len(d['y_all']), '| te_idx len =', len(te_iso),
      '| min/max =', te_iso.min(), te_iso.max(), '| dtype', te_iso.dtype)

p = os.path.join(R, 'ensemble_split_new.npz')
s = np.load(p)
print('\nensemble_split_new.npz keys:', s.files)
te_split = np.asarray(s['te_idx'])
print('  te_idx len =', len(te_split), '| min/max =', te_split.min(), te_split.max())
if 'cv_idx' in s.files:
    print('  cv_idx len =', len(np.asarray(s['cv_idx'])))

za = np.load(os.path.join(R, 'test_preds_A_8d6d.npz'), allow_pickle=True)
print('\ntest_preds_A_8d6d.npz keys:', list(za.files))
te_a = np.asarray(za['te_idx'])
print('  te_idx len =', len(te_a), '| min/max =', te_a.min(), te_a.max())

for name in ('test_preds_DimeNet.npz', 'test_preds_D_11d23d.npz', 'test_preds_DimeNet_fixed.npz'):
    q = os.path.join(R, name)
    if os.path.exists(q):
        z = np.load(q, allow_pickle=True)
        print('\n%s keys: %s' % (name, list(z.files)))
        for kk in z.files:
            v = z[kk]
            try:
                print('   ', kk, v.shape)
            except AttributeError:
                print('   ', kk, type(v))
        if 'te_idx' in z.files:
            t = np.asarray(z['te_idx'])
            print('  te_idx len =', len(t), '| min/max =', t.min(), t.max())
    else:
        print('\n%s 不存在' % name)

# 集合关系
s_iso = set(te_iso.tolist())
s_sp = set(te_split.tolist())
s_a = set(te_a.tolist())
print('\n集合关系：')
print('  A_te == split_new_te :', s_a == s_sp)
print('  A_te == iso_te       :', s_a == s_iso)
print('  split_new ⊂ iso      :', s_sp <= s_iso, '| iso-split =', sorted(s_iso - s_sp)[:5], '| split-iso =', sorted(s_sp - s_iso)[:5])
print('  A ⊂ iso              :', s_a <= s_iso, '| iso-A =', sorted(s_iso - s_a)[:5])

sp = os.path.join(R, 'server_package')
if os.path.isdir(sp):
    print('\nserver_package 目录：')
    for f in sorted(os.listdir(sp)):
        print('  ', f)
qq = os.path.join(R, 'README_dimenet_fixed.md')
print('\nREADME_dimenet_fixed.md 在根目录:', os.path.exists(qq))
