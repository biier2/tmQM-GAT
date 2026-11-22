# -*- coding: utf-8 -*-
"""Merge the DimeNet++-Q members and compare the ensemble with the published
DimeNet++ baseline on the same test split.

Inputs (defaults assume they sit in the working directory or its parent):

    --data          new_domain_iso.pt        (labels, atom lists, te_idx)
    --coords        schnet_domain_coords.npz (coordinates, per-complex lengths)
    --q-file        fixed_q.npz              (q_all, -1/0/+1)
    --ckpt-a        ensemble_checkpoint_DimeNetQ.pt (members trained in dir A)
    --ckpt-b        secondary checkpoint with the remaining member(s) (optional)
    --base-preds    preds/test_preds_DimeNet.npz of this repository

Outputs: q_vs_base_results.json and q_vs_base_preds.npz in the working directory.

Usage:
    python merge_eval_q.py --ckpt-a run_a/ensemble_checkpoint_DimeNetQ.pt \
        --ckpt-b run_b/ensemble_checkpoint_DimeNetQ.pt \
        --base-preds ../../../preds/test_preds_DimeNet.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score

sys.stdout.reconfigure(encoding='utf-8')

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus
from torch_geometric.nn.models.dimenet import triplets
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import scatter

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']


class DimeNetPlusPlusQ(DimeNetPlusPlus):
    """DimeNet++ with a learned embedding of the total charge added to the pooled output."""

    def __init__(self, *args, num_charge_states=3, **kwargs):
        hidden = kwargs.get('hidden_channels')
        out = kwargs.get('out_channels')
        super().__init__(*args, **kwargs)
        self.q_emb = torch.nn.Embedding(num_charge_states, hidden)
        self.q_out = torch.nn.Linear(hidden, out, bias=False)
        torch.nn.init.zeros_(self.q_out.weight)

    def forward(self, z, pos, batch=None, q=None):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=z.size(0))
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))
        out = scatter(P, batch, dim=0, reduce='sum') if batch is not None else P.sum(dim=0)
        if q is not None:
            out = out + self.q_out(self.q_emb(q))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='new_domain_iso.pt')
    ap.add_argument('--coords', default='schnet_domain_coords.npz')
    ap.add_argument('--q-file', default='fixed_q.npz')
    ap.add_argument('--ckpt-a', required=True)
    ap.add_argument('--ckpt-b', default='')
    ap.add_argument('--base-preds', required=True)
    ap.add_argument('--out-prefix', default='q_vs_base')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members = []
    for path in [p for p in (args.ckpt_a, args.ckpt_b) if p]:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        print(path, '->', [(m['fold'], m['seed']) for m in ck['models']])
        members.extend(ck['models'])
    members = sorted(members, key=lambda m: (m['fold'], m['seed']))
    print('merged members:', [(m['fold'], m['seed']) for m in members])

    D = torch.load(args.data, map_location='cpu', weights_only=False)
    y_all = D['y_all'].float()
    atoms = D['atoms_filt']
    te_idx = np.asarray(D['te_idx'])
    coords_np = np.load(args.coords)
    coords, lens = coords_np['coords'], coords_np['lens']
    q_all = np.load(args.q_file)['q_all']

    def pos_z(i):
        off = int(np.sum(lens[:i]))
        n = int(lens[i])
        return (torch.as_tensor(coords[off:off + n], dtype=torch.float32),
                torch.as_tensor(np.asarray(atoms[i]), dtype=torch.long))

    data = []
    for i in te_idx:
        p, z = pos_z(int(i))
        data.append(Data(z=z, pos=p, q=torch.tensor([int(q_all[int(i)]) + 1], dtype=torch.long)))
    loader = DataLoader(data, batch_size=8, shuffle=False)
    yt = y_all[te_idx].numpy().astype(np.float64)

    preds = []
    for m in members:
        model = DimeNetPlusPlusQ(hidden_channels=128, out_channels=8, num_blocks=4,
                                 int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
                                 num_spherical=7, num_radial=6, cutoff=5.0, max_num_neighbors=32,
                                 envelope_exponent=5, num_before_skip=1, num_after_skip=2,
                                 num_output_layers=3).to(device)
        model.load_state_dict(m['state'])
        model.eval()
        out = []
        with torch.no_grad():
            for d in loader:
                d = d.to(device)
                out.append(model(d.z, d.pos, d.batch, d.q.squeeze(-1)).float().cpu())
        preds.append(torch.cat(out).numpy() * m['y_std'].numpy() + m['y_mean'].numpy())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    P = np.stack(preds, 0)
    ens = P.mean(0)

    base = np.load(args.base_preds)
    pb = base['ens_mean'].astype(np.float64)
    assert np.allclose(base['y_true'], yt, atol=1e-4), 'y_true mismatch with base predictions'

    print('\n%-16s %9s %9s %9s | %9s %9s %9s'
          % ('property', 'base R2', 'Q R2', 'dR2', 'base MAE', 'Q MAE', 'dMAE'))
    rb, rq = [], []
    for i, t in enumerate(TARGETS):
        yy = yt[:, i]
        b2 = r2_score(yy, pb[:, i])
        q2 = r2_score(yy, ens[:, i])
        bm = mean_absolute_error(yy, pb[:, i])
        qm = mean_absolute_error(yy, ens[:, i])
        rb.append(b2)
        rq.append(q2)
        print('%-16s %9.4f %9.4f %+9.4f | %9.4g %9.4g %+9.3g'
              % (t, b2, q2, q2 - b2, bm, qm, qm - bm))
    print('%-16s %9.4f %9.4f %+9.4f' % ('mean (8)', np.mean(rb), np.mean(rq),
                                        np.mean(rq) - np.mean(rb)))
    print('%-16s %9.4f %9.4f %+9.4f' % ('intensive-5', np.mean(rb[:5]), np.mean(rq[:5]),
                                        np.mean(rq[:5]) - np.mean(rb[:5])))

    qt = q_all[te_idx]
    print('\ncharge counts in test:', dict(zip(*np.unique(qt, return_counts=True))))
    for name, idx in (('HL-Gap', 0), ('HOMO', 1), ('LUMO', 2), ('Dipole', 3)):
        print('%s by charge:' % name)
        for v in (-1, 0, 1):
            mm = qt == v
            b2 = r2_score(yt[mm, idx], pb[mm, idx])
            q2 = r2_score(yt[mm, idx], ens[mm, idx])
            print('  q=%+d n=%5d  base %.4f  Q %.4f  %+.4f' % (v, mm.sum(), b2, q2, q2 - b2))

    res = {'n_models': len(P), 'members': [[m['fold'], m['seed']] for m in members]}
    for i, t in enumerate(TARGETS):
        res[t] = {'base_r2': float(rb[i]), 'q_r2': float(rq[i]),
                  'base_mae': float(mean_absolute_error(yt[:, i], pb[:, i])),
                  'q_mae': float(mean_absolute_error(yt[:, i], ens[:, i]))}
    res['mean8'] = {'base': float(np.mean(rb)), 'q': float(np.mean(rq))}
    res['intensive5'] = {'base': float(np.mean(rb[:5])), 'q': float(np.mean(rq[:5]))}
    json.dump(res, open(args.out_prefix + '_results.json', 'w'), indent=2)
    np.savez(args.out_prefix + '_preds.npz', y_true=yt, base=pb, q=ens, members=P, te_idx=te_idx)
    print('\nsaved %s_results.json / %s_preds.npz' % (args.out_prefix, args.out_prefix))


if __name__ == '__main__':
    main()
