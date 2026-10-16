print('='*60)
print('Step 1: XYZ -> Distance Graph -> XYZ GNN (no SMILES needed)')
print('='*60)

import tarfile, os, glob

xyz_dir = 'BOSTMC/manyDFA/top100_divergence'
xyz_files = glob.glob(os.path.join(xyz_dir, '*.xyz')) if os.path.exists(xyz_dir) else []
if not xyz_files:
    os.makedirs(xyz_dir, exist_ok=True)
    try:
        with tarfile.open('BOSTMC.tar.gz', 'r:gz') as tar:
            members = [m for m in tar.getmembers() if 'top100_divergence' in m.name and m.name.endswith('.xyz')]
            for m in tqdm(members, desc='Extracting'):
                tar.extract(m, path='.')
        xyz_files = glob.glob(os.path.join(xyz_dir, '*.xyz'))
    except Exception as e:
        xyz_files = glob.glob('BOSTMC/**/*.xyz', recursive=True)

print(f'Found {len(xyz_files)} XYZ files')
print('df_clean rows:', len(df_clean), 'sample refcodes:', df_clean['refcode'].head(3).tolist())

if len(xyz_files) == 0:
    print('No XYZ files found.')
    graphs = []
    y_true = np.zeros((0, len(targets_ext)), dtype=np.float32)
else:
    # Improved XYZ-to-graph: compute degree + real distance RBF
    # Still missing: formal charge, aromaticity, Mayer BO (set to 0)
    graphs, y_true, matched_refs = [], [], []
    for xyz_path in tqdm(xyz_files, desc='XYZ->Graph'):
        refcode = os.path.basename(xyz_path).replace('.xyz', '')
        match = df_clean[df_clean['refcode'].str.strip() == refcode]
        if len(match) == 0:
            continue
        row = match.iloc[0]

        # Parse XYZ
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        symbols = []
        coords = []
        for line in lines[2:2+n_atoms]:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        coords = np.array(coords, dtype=np.float32)

        # Distance matrix
        diff = coords[np.newaxis,:,:] - coords[:,np.newaxis,:]
        dist_mat = np.sqrt((diff**2).sum(axis=2))

        # Build edges by covalent radii (1.3x tolerance)
        src, dst, edge_feats = [], [], []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                d = dist_mat[i,j]
                if d > 5.0:
                    continue
                ri = COVALENT_RADII.get(ATOM_TO_Z.get(symbols[i], 0), 0.7)
                rj = COVALENT_RADII.get(ATOM_TO_Z.get(symbols[j], 0), 0.7)
                if d < (ri + rj) * 1.3:
                    src.extend([i,j]); dst.extend([j,i])
                    za = ATOM_TO_Z.get(symbols[i], 0); zb = ATOM_TO_Z.get(symbols[j], 0)
                    is_ml = float((za>=21)!=(zb>=21))
                    feat = [0.0]*4  # no bond type info
                    feat.extend([0.0, is_ml])
                    for mu in rbf_c: feat.append(np.exp(-(d-mu)**2/rbf_sigma))
                    feat.append(0.0)  # BO unknown
                    edge_feats.extend([feat, feat])

        # Node features (11-dim): compute degree from connectivity
        degrees = [0]*n_atoms
        for s in src:
            degrees[s] += 1
        xf = []
        for s in symbols:
            z = ATOM_TO_Z.get(s, 0); pg = PERIOD_GROUP.get(z, (0,0))
            deg = degrees[len(xf)]
            xf.append([z, deg, 0, pg[0], pg[1], 0, 0, int(z>=21), 0, 0.0, 0.0])
        x = torch.tensor(xf, dtype=torch.float32)
        ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
        ea = torch.tensor(edge_feats, dtype=torch.float32) if edge_feats else torch.zeros(0,23)

        graphs.append({'x':x, 'edge_index':ei, 'edge_attr':ea, 'num_nodes':n_atoms})
        y_true.append([row[pbe0_cols[p]] for p in targets_ext])
        matched_refs.append(refcode)

    y_true = np.array(y_true, dtype=np.float32)
    print(f'Built {len(graphs)} graphs with matching PBE0 data')
