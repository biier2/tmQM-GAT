# -*- coding: utf-8 -*-
"""第三轮硬伤一(续): 2x2 degree x RBF 消融 (诊断变体 G/H)
G_7d6d  = A 去掉 degree 维 (7d/6d)          -> degree(拓扑配位数)本身的基线价值
H_7d22d = G + 16 RBF 距离 (7d/22d)           -> RBF 能否替代 degree 携带的配位数信息
对照已有: A(8d/6d 有degree无RBF) / B(8d/22d 有degree有RBF)
判据: H ~= B => RBF 完全替代 degree;  H ~= A => RBF 仅补回损失部分
--- 原说明 ---
第三轮硬伤一: WBO 节点特征的可加分解 (诊断变体 E/F)
E_9d6d  = A(8d) + node3[0]       即 WBO 有效配位数 nb_cnt     -> 测配位数的增量贡献
F_10d6d = A(8d) + node3[1],node3[2] 即 sum_BO + max_BO        -> 测键级大小的增量贡献
(已有 C_11d6d = A + node3[0,1,2] = E ∪ F, 用于验证可加性)
每变体 3-fold × 3-seed = 9 模型, 协议与 server_train_variants.py 逐字一致
数据: new_domain_iso.pt (服务器已有, 无需重传)
"""
import os, sys, time, json
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import Counter
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
from torch.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {torch.cuda.get_device_name(0)} VRAM {p.total_memory/1024**3:.1f}GB')

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
N_TARGETS = len(targets)
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10

VARIANTS = {
    'G_7d6d':  dict(npart=(), drop_deg=True, rbf=False, bo=False, tag='去 degree 7d/6d (诊断)'),
    'H_7d22d': dict(npart=(), drop_deg=True, rbf=True,  bo=False, tag='去 degree + RBF 7d/22d (诊断)'),
}

# ================= 数据加载 =================
DATA = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
y_all = DATA['y_all'].float()
N = y_all.size(0)
print(f'新域: N={N:,}')
for v, cfg in VARIANTS.items():
    print(f'  {v}: {cfg["tag"]}')

def build_variant_graphs(npart=(), drop_deg=False, rbf=True, bo=True):
    graphs = []
    for i in range(N):
        x = torch.as_tensor(DATA['node8s'][i], dtype=torch.float32)
        if drop_deg:
            # 8d = [Z, degree, charge, period, group, aromatic, nH, is_TM]
            # 去掉 index 1 (degree) -> is_TM 落到 index 6
            x = torch.cat([x[:, :1], x[:, 2:]], dim=1)
        if npart:
            n3 = torch.as_tensor(DATA['node3s'][i], dtype=torch.float32)
            x = torch.cat([x, n3[:, list(npart)]], dim=1)
        ea = DATA['e6_ls'][i]
        if rbf:
            ea = torch.cat([ea, DATA['rbf_ls'][i]], dim=1)
        if bo:
            ea = torch.cat([ea, DATA['bo1_ls'][i]], dim=1)
        graphs.append({'x': x, 'edge_index': DATA['ei_ls'][i], 'edge_attr': ea})
    return graphs

# ================= 划分 (同协议: 所有变体共用同一划分 → 完全配对) =================
idx_all = np.random.RandomState(42).permutation(N)
n_test = int(N * 0.15)
te_idx = idx_all[:n_test]
cv_idx = idx_all[n_test:]
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_splits = list(kf.split(cv_idx))
print(f'Test: {len(te_idx):,}  CV: {len(cv_idx):,}')
np.savez('ensemble_split_new.npz', te_idx=te_idx, cv_idx=cv_idx,
         fold_0_train=cv_idx[fold_splits[0][0]], fold_0_val=cv_idx[fold_splits[0][1]],
         fold_1_train=cv_idx[fold_splits[1][0]], fold_1_val=cv_idx[fold_splits[1][1]],
         fold_2_train=cv_idx[fold_splits[2][0]], fold_2_val=cv_idx[fold_splits[2][1]])

# ================= 金属权重 (原协议 sqrt 截断 [0.5,3.0]) =================
atoms_filt = DATA['atoms_filt']
mc = Counter()
for atoms in atoms_filt:
    for z in set(atoms.tolist()):
        if z >= 21: mc[z] += 1
total_c = len(atoms_filt)
weight_map = {z: max(0.5, min(3.0, (total_c / c) ** 0.5)) for z, c in mc.items()}
print(f'Metal weight map: {dict(sorted(weight_map.items()))}')
sample_weights = torch.ones(N)
for i, atoms in enumerate(atoms_filt):
    w = max((weight_map.get(z, 1.0) for z in set(atoms.tolist()) if z >= 21), default=1.0)
    sample_weights[i] = w
sample_weights = sample_weights / sample_weights.mean()
print(f'Sample weights: mean={sample_weights.mean():.3f}, max={sample_weights.max():.2f}')

# ================= 模型 (与 notebook 逐字一致) =================
def segment_softmax(scores, dst, Nn):
    scores_f32 = scores.float()
    max_v = torch.full((Nn, scores_f32.size(1)), float('-inf'), device=scores_f32.device, dtype=torch.float32)
    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), scores_f32, reduce='amax', include_self=False)
    exp_s = torch.exp(scores_f32 - max_v[dst])
    sum_v = torch.zeros(Nn, scores_f32.size(1), device=scores_f32.device, dtype=torch.float32)
    sum_v.scatter_add_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), exp_s)
    return (exp_s / (sum_v[dst] + 1e-8)).to(scores.dtype)

class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim_per_head, edge_dim=0, heads=4):
        super().__init__(); self.heads, self.out_dim = heads, out_dim_per_head
        total_dim = out_dim_per_head * heads
        self.node_lin = nn.Linear(in_dim, total_dim, bias=False)
        self.attn_src = nn.Linear(total_dim, heads, bias=False)
        self.attn_dst = nn.Linear(total_dim, heads, bias=False)
        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim > 0 else None
        self.act = nn.ReLU()
    def forward(self, x, ei, ea=None):
        src, dst = ei[0], ei[1]; Nn = x.size(0); h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0: attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, Nn)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(Nn, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out)

def scatter_mean_per_graph(h, batch, n_graphs):
    out = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
    cnt = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
    cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)

def scatter_max_per_graph(h, batch, n_graphs):
    out = torch.full((n_graphs, h.size(1)), float('-inf'), device=h.device, dtype=h.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce='amax', include_self=False)
    return out

class XYZ_GNN(nn.Module):
    def __init__(self, node_dim=11, edge_dim=23, hidden=HIDDEN, n_layers=N_LAYERS,
                 output_dim=N_TARGETS, dropout=DROPOUT, heads=GAT_HEADS, tm_idx=7):
        super().__init__(); self.heads = heads; self.tm_idx = tm_idx
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            mm = (x[:, self.tm_idx] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        mm = (x[:, self.tm_idx] > 0.5)
        mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

# ================= 训练 (与 notebook cell5 逐字一致) =================
transfer_stream = torch.cuda.Stream()

def train_one_model(graphs, y_all_gpu, sw_gpu, train_idx, val_idx, seed, fold_name,
                    node_dim, edge_dim, tm_idx=7):
    torch.manual_seed(seed); np.random.seed(seed)
    g_x = [g['x'] for g in graphs]
    g_ei = [g['edge_index'] for g in graphs]
    g_ea = [g['edge_attr'] for g in graphs]

    def collate_fast(indices):
        x_list = [g_x[i] for i in indices]
        ei_list = [g_ei[i] for i in indices]
        ea_list = [g_ea[i] for i in indices]
        batch_list = []
        offset = 0
        for k, x in enumerate(x_list):
            n = x.size(0)
            batch_list.append(torch.full((n,), k, dtype=torch.long))
            ei_list[k] = ei_list[k] + offset
            offset += n
        x_cat = torch.cat(x_list, dim=0)
        ei_cat = torch.cat(ei_list, dim=1)
        ea_cat = torch.cat(ea_list, dim=0)
        b_cat = torch.cat(batch_list, dim=0)
        with torch.cuda.stream(transfer_stream):
            batched = {'x': x_cat.to(device, non_blocking=True),
                       'edge_index': ei_cat.to(device, non_blocking=True),
                       'edge_attr': ea_cat.to(device, non_blocking=True)}
            batch_idx = b_cat.to(device, non_blocking=True)
        return batched, batch_idx

    model = XYZ_GNN(node_dim=node_dim, edge_dim=edge_dim, tm_idx=tm_idx).to(device)
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    crit = nn.MSELoss(reduction='none')
    scaler = GradScaler('cuda')

    y_tr = y_all_gpu[train_idx]
    y_mean_local = y_tr.mean(dim=0)
    y_std_local = y_tr.std(dim=0).clamp(min=1e-8)
    y_norm = (y_all_gpu - y_mean_local) / y_std_local

    bs = 256
    best_va = float('inf'); best_state = None; patience = 0
    epochs = 200; early_stop = 40
    n_tr = len(train_idx); n_va = len(val_idx)
    print(f'  [{fold_name}] epochs={epochs} train={n_tr} val={n_va} bs={bs}', flush=True)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        tr_loss = 0.0; nb = 0
        first_pos = perm[:bs]
        first_bidx = train_idx[first_pos].tolist()
        next_batch = collate_fast(first_bidx)
        next_by = y_norm[first_bidx]
        next_w = sw_gpu[first_bidx]

        for s in range(0, n_tr, bs):
            batched, batch_idx = next_batch
            by = next_by; w = next_w
            next_s = s + bs
            if next_s < n_tr:
                next_pos = perm[next_s:next_s+bs]
                next_bidx = train_idx[next_pos].tolist()
                next_batch = collate_fast(next_bidx)
                next_by = y_norm[next_bidx]
                next_w = sw_gpu[next_bidx]
            else:
                next_batch = None
            torch.cuda.current_stream().wait_stream(transfer_stream)
            opt.zero_grad()
            with autocast('cuda'):
                preds = model(batched, batch_idx)
                loss = (crit(preds, by).mean(dim=1) * w).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr_loss += loss.item(); nb += 1
        sched.step(ep + tr_loss / max(1, nb))

        model.eval()
        with torch.no_grad():
            va_preds = []
            for s in range(0, n_va, bs):
                bidx = val_idx[s:s+bs].tolist()
                batched, batch_idx = collate_fast(bidx)
                with autocast('cuda'):
                    va_preds.append(model(batched, batch_idx).float())
            vp = torch.cat(va_preds, dim=0)
            va_loss = nn.MSELoss()(vp, y_norm[val_idx])
        if va_loss < best_va:
            best_va = va_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= early_stop:
            print(f'  [{fold_name}] Early stop @ ep {ep+1}', flush=True)
            break
        if (ep+1) % 30 == 0:
            print(f'  [{fold_name}] Ep {ep+1}: tr={tr_loss/max(1,nb):.4f} va={va_loss.item():.4f} pat={patience}', flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vp_full = []
        for s in range(0, n_va, bs):
            bidx = val_idx[s:s+bs].tolist()
            batched, batch_idx = collate_fast(bidx)
            with autocast('cuda'):
                vp_full.append(model(batched, batch_idx).detach().cpu().float())
        vp_full = torch.cat(vp_full, dim=0)
    vt_np = y_all[val_idx].numpy()
    vp_orig = vp_full.numpy() * y_std_local.cpu().numpy() + y_mean_local.cpu().numpy()
    val_r2s = [r2_score(vt_np[:, i], vp_orig[:, i]) for i in range(N_TARGETS)]
    print(f'  [{fold_name}] Done: best_va={best_va:.4f}, val Mean R2={np.mean(val_r2s):.4f}')
    del model; torch.cuda.empty_cache()
    return best_state, y_mean_local.cpu(), y_std_local.cpu(), val_r2s

# ================= 主循环: 变体 × 3-fold × 3-seed =================
SEEDS = [42, 123, 456]
y_all_gpu = y_all.to(device)
sw_gpu = sample_weights.to(device)
t_start = time.time()

for VNAME, VCFG in VARIANTS.items():
    node_dim = 8 - (1 if VCFG.get('drop_deg') else 0) + len(VCFG['npart'])
    tm_idx = 7 - (1 if VCFG.get('drop_deg') else 0)
    edge_dim = 6 + (16 if VCFG['rbf'] else 0) + (1 if VCFG['bo'] else 0)
    ckpt_path = f'ensemble_checkpoint_{VNAME}.pt'
    print(f'\n{"="*70}\nVARIANT {VNAME} ({VCFG["tag"]}) node={node_dim} edge={edge_dim}\n{"="*70}', flush=True)
    graphs = build_variant_graphs(npart=VCFG['npart'], drop_deg=VCFG.get('drop_deg', False), rbf=VCFG['rbf'], bo=VCFG['bo'])
    print(f'  graphs built: {len(graphs):,}', flush=True)

    all_models, results_log = [], []
    completed = set()
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        all_models, results_log = ck['models'], ck.get('log', [])
        for m in all_models: completed.add((m['fold'], m['seed']))
        print(f'  Restored {len(all_models)} models: {sorted(completed)}')

    for fold_idx in range(3):
        tr_abs = cv_idx[fold_splits[fold_idx][0]]; va_abs = cv_idx[fold_splits[fold_idx][1]]
        for seed in SEEDS:
            if (fold_idx, seed) in completed:
                print(f'  SKIP Fold{fold_idx}_Seed{seed}'); continue
            name = f'Fold{fold_idx}_Seed{seed}'
            t0 = time.time()
            print(f'  Training {name} (train={len(tr_abs):,} val={len(va_abs):,})...', flush=True)
            try:
                state, ym, ys, val_r2s = train_one_model(graphs, y_all_gpu, sw_gpu,
                                                         tr_abs, va_abs, seed, name, node_dim, edge_dim, tm_idx)
                el = (time.time() - t0) / 60
                all_models.append({'state': state, 'y_mean': ym, 'y_std': ys,
                                   'fold': fold_idx, 'seed': seed, 'val_r2s': val_r2s})
                results_log.append({'fold': fold_idx, 'seed': seed,
                                    'val_mean_r2': float(np.mean(val_r2s)), 'val_r2s': val_r2s,
                                    'time_min': round(el, 1)})
                completed.add((fold_idx, seed))
                print(f'  Done {name} in {el:.1f}min, val Mean R2={np.mean(val_r2s):.4f}')
            except Exception as e:
                import traceback; traceback.print_exc()
                results_log.append({'fold': fold_idx, 'seed': seed, 'error': str(e)})
        torch.save({'models': all_models, 'log': results_log}, ckpt_path)
        print(f'  checkpoint saved ({len(all_models)}/9)')

    # ---- 测试集 ensemble 评估 ----
    print(f'  Test eval for {VNAME}...', flush=True)
    g_x = [g['x'] for g in graphs]; g_ei = [g['edge_index'] for g in graphs]
    g_ea = [g['edge_attr'] for g in graphs]
    def collate_test(indices):
        x_list = [g_x[i] for i in indices]; ei_list = [g_ei[i] for i in indices]
        ea_list = [g_ea[i] for i in indices]; batch_list = []; offset = 0
        for k, x in enumerate(x_list):
            n = x.size(0); batch_list.append(torch.full((n,), k, dtype=torch.long))
            ei_list[k] = ei_list[k] + offset; offset += n
        return {'x': torch.cat(x_list, 0).to(device), 'edge_index': torch.cat(ei_list, 1).to(device),
                'edge_attr': torch.cat(ea_list, 0).to(device)}, torch.cat(batch_list, 0).to(device)

    all_test_preds = []
    for mdata in all_models:
        model = XYZ_GNN(node_dim=node_dim, edge_dim=edge_dim, tm_idx=tm_idx).to(device)
        state = {k.replace('_orig_mod.', ''): v for k, v in mdata['state'].items()}
        model.load_state_dict(state); model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(te_idx), 64):
                bidx = te_idx[s:s+64].tolist()
                batched, batch_idx = collate_test(bidx)
                with autocast('cuda'):
                    p = model(batched, batch_idx).float().cpu()
                preds.append(p)
        preds = torch.cat(preds, 0)
        all_test_preds.append((preds * mdata['y_std'] + mdata['y_mean']).numpy())
        del model; torch.cuda.empty_cache()
    all_preds = np.stack(all_test_preds, 0)
    ens_mean = all_preds.mean(0)
    yt = y_all[te_idx].numpy()
    res = {'variant': VNAME, 'node_dim': node_dim, 'edge_dim': edge_dim}
    nM = len(all_models)
    for i, prop in enumerate(targets):
        res[prop] = {'ens_r2': float(r2_score(yt[:, i], ens_mean[:, i])),
                     'mean_r2': float(np.mean([r2_score(yt[:, i], all_preds[m, :, i]) for m in range(nM)])),
                     'std_r2': float(np.std([r2_score(yt[:, i], all_preds[m, :, i]) for m in range(nM)])),
                     'ens_mae': float(mean_absolute_error(yt[:, i], ens_mean[:, i]))}
    res['mean_ens_r2'] = float(np.mean([res[p]['ens_r2'] for p in targets]))
    res['cv_val_mean'] = float(np.mean([r['val_mean_r2'] for r in results_log if 'val_mean_r2' in r]))
    np.savez(f'test_preds_{VNAME}.npz', preds=all_preds, ens_mean=ens_mean, y_true=yt, te_idx=te_idx)
    with open(f'results_{VNAME}.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f'  {VNAME} mean ensemble R2 = {res["mean_ens_r2"]:.4f}')
    print(json.dumps(res, indent=2)[:800])

print(f'\nALL DONE in {(time.time()-t_start)/60:.0f} min')
