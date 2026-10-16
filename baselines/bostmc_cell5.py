print('='*60)
print('Step 2: XYZ GNN Inference on BOSTMC (distance graph)')
print('='*60)

if len(graphs) == 0:
    print('ERROR: No graphs to run inference on.')
else:
    # Load XYZ GNN checkpoint
    import torch.nn as nn

    HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12

    class XYZ_GNN_Eval(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(11, HIDDEN)
            self.convs = nn.ModuleList([GATConv(HIDDEN, HIDDEN//GAT_HEADS, 23, heads=GAT_HEADS) for _ in range(N_LAYERS)])
            self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
            self.drop = nn.Dropout(0.10)
            self.readout = nn.Sequential(
                nn.Linear(HIDDEN*3, HIDDEN), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(HIDDEN, HIDDEN//2), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(HIDDEN//2, 8))
        def forward(self, g, batch=None):
            x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
            h = self.embed(x)
            for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
            if batch is None:
                mm = (x[:,7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
                return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
            n_graphs = int(batch.max().item()) + 1
            mean_h = scatter_mean_per_graph(h, batch, n_graphs); max_h = scatter_max_per_graph(h, batch, n_graphs)
            mm = (x[:,7] > 0.5)
            mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

    model = XYZ_GNN_Eval().to(device)
    ckpt = 'xyz_best.pt'
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f'Loaded {ckpt}')
    else:
        print(f'{ckpt} not found!')
        exit(1)
    model.eval()

    bs = 32
    y_mean = np.array([0.108504, -0.197593, -0.089089, 5.799450, 0.162882, -2987.303526, -0.142228, 397.124993], dtype=np.float32)
    y_std  = np.array([0.033575, 0.053825, 0.054517, 3.927489, 0.782914, 1634.083288, 0.070115, 152.460083], dtype=np.float32)

    preds_list = []
    with torch.no_grad():
        for s in tqdm(range(0, len(graphs), bs), desc='XYZ GNN'):
            bg = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.items()} for g in graphs[s:s+bs]]
            batched, batch_idx = collate_graphs(bg)
            pn = model(batched, batch_idx).cpu().numpy()
            preds_list.append(pn * y_std + y_mean)
    preds_all = np.concatenate(preds_list, axis=0)

    target_idx = [0, 1, 2, 3, 4, 5]

    print(f"\n{'='*80}")
    print(f"BOSTMC VALIDATION (N={len(graphs)}, XYZ->distance graph, 0% CSD overlap)")
    print(f"{'='*80}")
    print(f"{'Property':16s} {'Pearson r':>10s} {'MAE':>12s} {'R2':>10s}")
    print('-' * 52)

    results = []
    for j, prop in enumerate(targets_ext):
        yt = y_true[:, j]
        yp = preds_all[:, target_idx[j]]
        r_val, _ = pearsonr(yt, yp)
        mae_val = mean_absolute_error(yt, yp)
        r2_val = r2_score(yt, yp)
        print(f'{prop:16s} {r_val:10.4f} {mae_val:12.4f} {r2_val:10.4f}')
        results.append({'Property': prop, 'Pearson_r': r_val, 'MAE': mae_val, 'R2': r2_val})

    res_df = pd.DataFrame(results)
    mean_r = res_df['Pearson_r'].mean()
    print(f'\nMean Pearson r: {mean_r:.4f}')
    res_df.to_csv('bostmc_validation_results.csv', index=False)
    print('Saved to bostmc_validation_results.csv')

    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    for j, prop in enumerate(targets_ext):
        ax = axes[j]
        yt = y_true[:, j]
        yp = preds_all[:, target_idx[j]]
        r_val, _ = pearsonr(yt, yp)
        ax.scatter(yt, yp, alpha=0.4, s=10, c='#e74c3c', edgecolors='none')
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel(f'PBE0 {prop}')
        ax.set_ylabel(f'Predicted {prop}')
        ax.set_title(f'{prop} (r={r_val:.3f})')
    plt.suptitle('BOSTMC Validation: XYZ GNN on Independent Dataset (0% CSD Overlap)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
