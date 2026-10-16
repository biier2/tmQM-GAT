"""Evaluate SchNet checkpoint on test set"""
import json, torch, torch.nn as nn, numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Load model definition
N_RBF = 20; HIDDEN = 128; N_LAYERS = 4; N_ATOM_TYPES = 85
device = torch.device('cuda')

class CFConv(nn.Module):
    def __init__(self, hidden, n_rbf):
        super().__init__()
        self.filter_net = nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
    def forward(self, h, ei, rbf):
        src, dst = ei[0], ei[1]; N = h.size(0)
        W = self.filter_net(rbf); msg = h[src] * W
        out = torch.zeros(N, h.size(1), device=h.device, dtype=h.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return out

def collate_schnet(batch_list):
    z_list, pos_list, ei_list, rbf_list, batch_list2 = [], [], [], [], []
    offset = 0
    for i, g in enumerate(batch_list):
        n = g['z'].size(0)
        z_list.append(g['z']); pos_list.append(g['pos'])
        ei_list.append(g['edge_index'] + offset); rbf_list.append(g['edge_rbf'])
        batch_list2.append(torch.full((n,), i, dtype=torch.long, device=g['z'].device))
        offset += n
    return {'z': torch.cat(z_list), 'pos': torch.cat(pos_list), 'edge_index': torch.cat(ei_list, dim=1), 'edge_rbf': torch.cat(rbf_list, dim=0), 'batch': torch.cat(batch_list2), 'n_graphs': len(batch_list)}

class SchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(N_ATOM_TYPES, HIDDEN, padding_idx=0)
        self.interactions = nn.ModuleList([CFConv(HIDDEN, N_RBF) for _ in range(N_LAYERS)])
        self.atomwise = nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN)) for _ in range(N_LAYERS)])
        self.scalar_head = nn.Sequential(nn.Linear(HIDDEN*2, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN//2), nn.SiLU(), nn.Linear(HIDDEN//2, 8))
        self.charge_head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN//2), nn.SiLU(), nn.Linear(HIDDEN//2, 1))
    def forward(self, g, batch_mode=True):
        z, pos, ei, rbf = g['z'], g['pos'], g['edge_index'], g['edge_rbf']
        batch, n_graphs = g['batch'], g['n_graphs']
        h = self.embed(z)
        for inter, atom in zip(self.interactions, self.atomwise):
            h = h + atom(inter(h, ei, rbf))
        mean_h = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        mean_h.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
        cnt = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
        cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
        mean_h = mean_h / cnt.clamp(min=1).unsqueeze(-1)
        sum_h = mean_h * cnt.unsqueeze(-1)
        h_pool = torch.cat([mean_h, sum_h], dim=1)
        scalar_pred = self.scalar_head(h_pool)
        q = self.charge_head(h).squeeze(-1)
        dipole_mags = torch.zeros(n_graphs, device=h.device)
        for gi in range(n_graphs):
            mask = batch == gi
            qi, pi = q[mask], pos[mask]
            q_abs_sum = qi.abs().sum() + 1e-8
            r_center = (pi * qi.abs().unsqueeze(-1)).sum(dim=0) / q_abs_sum
            dipole_mags[gi] = (qi.unsqueeze(-1) * (pi - r_center)).sum(dim=0).norm()
        scalar_pred[:, 3] = dipole_mags
        return scalar_pred

# Load model
model = SchNet().to(device)
model.load_state_dict(torch.load('schnet_best.pt', map_location=device))
model.eval()
print('Model loaded')

# Extract data from notebook
nb = json.load(open('tmQM_SchNet.ipynb'))
namespace = {}
for c in nb['cells']:
    src = ''.join(c['source'])
    if 'te_idx' in src and 'va_idx' in src and 'tr_idx' in src:
        exec(src, namespace)
        print(f'Found split indices')
        break

te_idx = namespace.get('te_idx')
y_std = namespace.get('y_std')
y_mean = namespace.get('y_mean')
y_all = namespace.get('y_all')
targets = namespace.get('targets')

if te_idx is None:
    print('ERROR: Could not extract te_idx from notebook. Run cell 5 first.')
    exit(1)

# Load test graphs from notebook (re-run cell 5 first!)
print(f'Test set: {len(te_idx)} complexes')
print('Loading test graphs...')
# We need graphs to be available. They are stored in memory during notebook runs.
# This standalone script cannot access notebook memory.
# Alternative: load graphs from disk (they need to be saved)

print('Graphs are only in notebook memory. Run this inside the notebook instead:')
print('Open tmQM_SchNet.ipynb in Jupyter and execute Cell 8 manually.')
