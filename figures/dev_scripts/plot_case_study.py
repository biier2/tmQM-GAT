"""Figure 7: Case study — predicted vs DFT HL-Gap for 5 catalysts"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Data: (name, pred_eV, dft_eV)
catalysts = [
    ('Zr-cat3', 3.99, 3.41),
    ('Co-cat1', 1.92, 2.87),
    ('Fe-cat1', 1.91, 2.82),
    ('Pd-cat1', 1.72, 2.78),
    ('Ni-cat3', 1.79, 1.25),
]

names = [c[0] for c in catalysts]
preds = [c[1] for c in catalysts]
dfts  = [c[2] for c in catalysts]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, preds, width, label='Predicted (Dual-Model Scorer)', color='#3498DB', edgecolor='white')
bars2 = ax.bar(x + width/2, dfts, width, label='DFT (TPSSh-D3BJ/def2-SVP)', color='#E74C3C', edgecolor='white')

# Annotate values
for bar, val in zip(bars1, preds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{val:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, dfts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{val:.2f}',
            ha='center', va='bottom', fontsize=10)

# Annotate |delta| above the taller bar
for i in range(len(names)):
    delta = abs(preds[i] - dfts[i])
    y_top = max(preds[i], dfts[i])
    ax.text(x[i], y_top + 0.55, f'|Δ|={delta:.2f} eV',
            ha='center', fontsize=8, color='#333', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel('HL-Gap (eV)', fontsize=13)
ax.set_title('Case Study: Predicted vs DFT HL-Gap', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, max(max(preds), max(dfts)) * 1.55)
ax.grid(axis='y', alpha=0.3)

# Add MAE annotation
mae = np.mean([abs(preds[i]-dfts[i]) for i in range(len(names))])
ax.text(0.98, 0.95, f'MAE = {mae:.2f} eV\n(0.030 Ha)', transform=ax.transAxes,
        ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figure7_case_study.png', dpi=300, bbox_inches='tight', facecolor='white')
print('Saved figure7_case_study.png')
