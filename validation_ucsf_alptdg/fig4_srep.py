"""Figure 4 (Scientific Reports version): annotated growth, time probe and spatial ranking."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'axes.linewidth': 1.0, 'xtick.major.width': 1.0,
                     'ytick.major.width': 1.0, 'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.width': 1.0,
                     'ytick.minor.width': 1.0, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8, 'axes.axisbelow': True,
                     'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'axes.edgecolor': '#333333', 'xtick.color': '#333333', 'ytick.color': '#333333',
                     'axes.labelcolor': '#111111'})
BLUE, ORANGE, INK, GRID, REF, GREY = '#1F6FB2', '#D55E00', '#111111', '#EBEBEB', '#666666', '#5F5F5F'
mm = 1 / 25.4
B = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(B, 'data/ucsf/infer_flair/per_patient.csv')).drop_duplicates('id').set_index('id').sort_index()
cl = pd.read_csv(os.path.join(B, 'data/ucsf/clinical_per_patient.csv')).set_index('id').loc[d.index]
s = pd.read_csv(os.path.join(B, 'data/ucsf/spatial_flair/per_patient.csv')).drop_duplicates('id').set_index('id').sort_index()

def style(ax):
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=1.0, zorder=0)

fig, (a2, a1, a3) = plt.subplots(1, 3, figsize=(174 * mm, 62 * mm), gridspec_kw={'width_ratios': [1.05, 0.9, 1.05]})
# b: Dice difference by annotated change
diff = d.tumor_true_dice - d.tumor_cf_dice
rng = np.random.default_rng(1)
groups = [(~cl.annotated_increase.astype(bool), 'No increase\nannotated'), (cl.annotated_increase.astype(bool), 'Increase\nannotated')]
a1.axhline(0, color=REF, lw=1.0, zorder=1)
import json
CE = json.load(open(os.path.join(B, 'data/ucsf/clinical_endpoints.json')))['groups']   # stored CIs (clinical_srep.py)
for k, ((m, lab), g) in enumerate(zip(groups, ['no_inc', 'inc'])):
    y = diff[m].values; x = k + rng.uniform(-0.17, 0.17, len(y))
    a1.scatter(x, y, s=15, marker='o', facecolor=BLUE, edgecolor='white', linewidth=1.0, zorder=3, label='Patients' if k == 0 else None)
    lo, hi = CE[g]['diff_ci']; assert abs(CE[g]['diff'] - y.mean()) < 1e-9
    eb = a1.errorbar(k + 0.3, y.mean(), yerr=[[y.mean() - lo], [hi - y.mean()]], fmt='o', ms=3.2, color=INK, lw=1.0, capsize=2, capthick=1.0, zorder=4)
a1.set_xticks([0, 1]); a1.set_xticklabels([f'{lab}\n(n = {int(m.sum())})' for m, lab in groups])
a1.set_xlim(-0.5, 1.6); a1.set_ylabel('Dice, model − no-change reference')
h, l = a1.get_legend_handles_labels(); a1.legend(h + [eb], l + ['Mean (95% CI)'], loc='lower right', frameon=False,
                                                 fontsize=6.4, handletextpad=0.3, borderaxespad=0.2)
ymax = float(np.nanmax(np.abs(diff))) * 1.08; a1.set_ylim(-ymax, max(0.12, float(diff.max()) * 1.2))
# a: time probe
t = s.pred_new_t / 1000; t0 = s.pred_new_t0 / 1000
lo_, hi_ = 0.3, float(max(t.max(), t0.max())) * 1.6
a2.plot([lo_, hi_], [lo_, hi_], ls=(0, (3, 2)), lw=1.0, color=REF, zorder=1)
a2.scatter(t.clip(lower=lo_), t0.clip(lower=lo_), s=16, marker='o', facecolor=BLUE, edgecolor='white', linewidth=1.0, zorder=3)
a2.set_xscale('log'); a2.set_yscale('log'); a2.set_xlim(lo_, hi_); a2.set_ylim(lo_, hi_); a2.set_aspect('equal')
a2.set_xlabel('New volume, actual interval, cm$^3$'); a2.set_ylabel('New volume, interval set to zero, cm$^3$')
share = s.pred_new_t0.sum() / s.pred_new_t.sum()
a2.text(0.97, 0.04, f'Pooled: {100 * share:.0f}% of the\nvolume for the actual interval', transform=a2.transAxes, ha='right',
        va='bottom', fontsize=6.0, color=INK)
a2.text(0.04, 0.96, 'Identity', transform=a2.transAxes, ha='left', va='top', fontsize=6.0, color=REF)
# c: AUC distance vs model
xa, ya = s.auc_proximity, s.auc_model_true; ok = ~(xa.isna() | ya.isna())
lo3 = float(np.floor(min(xa[ok].min(), ya[ok].min()) * 20) / 20) - 0.02
a3.plot([lo3, 1], [lo3, 1], ls=(0, (3, 2)), lw=1.0, color=REF, zorder=1)
a3.scatter(xa[ok], ya[ok], s=16, marker='o', facecolor=BLUE, edgecolor='white', linewidth=1.0, zorder=3)
a3.set_xlim(lo3, 1); a3.set_ylim(lo3, 1); a3.set_aspect('equal')
a3.set_xlabel('AUC, distance from baseline tumor'); a3.set_ylabel('AUC, model probability')
a3.text(0.97, 0.04, f'Distance higher:\nn = {int((xa[ok] > ya[ok]).sum())} of {int(ok.sum())}', transform=a3.transAxes,
        ha='right', va='bottom', fontsize=6.0, color=INK)
for ax in (a1, a2, a3): style(ax)
for ax, l, x in zip((a2, a1, a3), 'abc', (-0.26, -0.3, -0.26)):
    ax.text(x, 1.03, l, transform=ax.transAxes, fontsize=7, fontweight='bold', va='bottom')
fig.tight_layout(pad=0.4, w_pad=1.6)
for ext in ['pdf', 'eps']: fig.savefig(os.path.join(B, f'figs/Fig4_srep.{ext}'))
fig.savefig(os.path.join(B, 'figs/Fig4_srep.png'), dpi=600)
fig.savefig(os.path.join(B, 'figs/Fig4_srep.tif'), dpi=1200, pil_kwargs={'compression': 'tiff_lzw'})
print('Fig4: n spatial', len(s), '| t0 share', round(share, 3), '| distance higher', int((xa[ok] > ya[ok]).sum()), 'of', int(ok.sum()),
      '| diff range', round(diff.min(), 3), round(diff.max(), 3), '| t range', round(t.min(), 2), round(t.max(), 1))
