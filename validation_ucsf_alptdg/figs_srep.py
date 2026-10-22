"""Figures for the Scientific Reports version: Fig. 2 (tumor masks), Fig. S1 (flow), Fig. S2 (Bland-Altman),
Fig. S3 (calibration)."""
import os, numpy as np, pandas as pd, matplotlib, json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'axes.linewidth': 1.0, 'xtick.major.width': 1.0,
                     'ytick.major.width': 1.0, 'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.width': 1.0,
                     'ytick.minor.width': 1.0, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8, 'axes.axisbelow': True,
                     'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'axes.edgecolor': '#333333', 'xtick.color': '#333333', 'ytick.color': '#333333',
                     'axes.labelcolor': '#111111'})
BLUE, ORANGE, INK, GRID, REF, GREY = '#1F6FB2', '#D55E00', '#111111', '#EBEBEB', '#666666', '#5F5F5F'
mm = 1 / 25.4
T = 'tumor'
RUN = os.environ.get('UCSF_RUN', 'infer_flair')
d = pd.read_csv(f'data/ucsf/{RUN}/per_patient.csv').drop_duplicates('id').set_index('id')
fv = pd.read_excel('data/new/feat_val.xlsx'); fv['pid'] = fv.id.str[8:].astype(int); fv = fv.set_index('pid')
d = d.join(fv[['cortex']])
bl = d[f'{T}_cf_vol_pred'] / 1000; fu = d[f'{T}_cf_vol_gt'] / 1000

def style(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=1.0, zorder=0)

# ---------------- Fig. 2 ----------------
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(174 * mm, 62 * mm), gridspec_kw={'width_ratios': [1, 1.12, 1.12]})
c = d[f'{T}_cf_dice']; mdl = d[f'{T}_true_dice']
a1.plot([0, 1], [0, 1], ls=(0, (3, 2)), lw=1.0, color=REF, zorder=1)
a1.scatter(c, mdl, s=16, marker='o', facecolor=BLUE, edgecolor='white', linewidth=1.0, zorder=3)
a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.set_aspect('equal')
a1.set_xlabel('Dice, no-change reference'); a1.set_ylabel('Dice, conditioned model')
a1.text(0.97, 0.04, f'Higher with model:\nn = {(mdl > c).sum()} of {len(d)}', ha='right', va='bottom', fontsize=6.2, color=INK)
obs = fu - bl; pch = d[f'{T}_true_vol_pred'] / 1000 - bl
a2.axhline(0, color=REF, lw=1.2, zorder=1); a2.axvline(0, color=GRID, lw=1.0, zorder=0)
x0, x1 = np.floor(obs.min() / 10) * 10 - 2, np.ceil(obs.max() / 10) * 10 + 2
y0, y1 = min(np.floor(pch.min() / 10) * 10, -10) - 2, max(np.ceil(pch.max() / 10) * 10, 20) + 2
lo, hi = max(x0, y0), min(x1, y1)
a2.plot([lo, hi], [lo, hi], ls=(0, (3, 2)), lw=1.0, color=REF, zorder=1)
a2.scatter(obs, pch, s=16, marker='o', facecolor=BLUE, edgecolor='white', linewidth=1.0, zorder=3)
a2.set_xlim(x0, x1); a2.set_ylim(y0, y1)
a2.set_xlabel('Observed tumor-volume change, cm$^3$'); a2.set_ylabel('Predicted tumor-volume change, cm$^3$')
a2.text(x1 - 0.01 * (x1 - x0), -0.012 * (y1 - y0), 'No-change reference', ha='right', va='top', fontsize=6.0, color=REF)
a2.text(hi - 0.03 * (x1 - x0), hi - 0.04 * (y1 - y0), 'Identity', ha='right', va='top', fontsize=6.0, color=REF)
print('Fig2b limits', x0, x1, y0, y1)
iso = d.cortex == 'Isocortex'
dv = (d[f'{T}_cortex_swap_vol_pred'] - d[f'{T}_true_vol_pred']) / 1000
dd = d[f'{T}_cortex_swap_dice'] - d[f'{T}_true_dice']
a3.axhline(0, color=REF, lw=1.0, zorder=1); a3.axvline(0, color=REF, lw=1.0, zorder=1)
a3.scatter(dv[iso], dd[iso], s=16, marker='o', facecolor=BLUE, edgecolor='white', linewidth=1.0, zorder=3,
           label=f'Isocortical → mesocortical (n = {iso.sum()})')
a3.scatter(dv[~iso], dd[~iso], s=22, marker='^', facecolor=ORANGE, edgecolor='white', linewidth=1.0, zorder=4,
           label=f'Other → isocortical (n = {(~iso).sum()})')
a3.set_xlabel('Change in predicted volume,\nswapped vs correct label, cm$^3$'); a3.set_ylabel('Change in Dice, swapped vs correct label')
a3.legend(loc='upper right', frameon=False, fontsize=5.8, handletextpad=0.2, borderaxespad=0.1)
ymax = float(np.nanmax(np.abs(dd))) * 1.1; a3.set_ylim(-ymax, ymax)
for ax in (a1, a2, a3): style(ax)
for ax, l, x in zip((a1, a2, a3), 'abc', (-0.26, -0.2, -0.24)):
    ax.text(x, 1.03, l, transform=ax.transAxes, fontsize=7, fontweight='bold', va='bottom')
fig.tight_layout(pad=0.4, w_pad=1.4)
for ext in ['pdf', 'eps']: fig.savefig(f'figs/Fig2_srep.{ext}')
fig.savefig('figs/Fig2_srep.png', dpi=600); fig.savefig('figs/Fig2_srep.tif', dpi=1200, pil_kwargs={'compression': 'tiff_lzw'})
print('Fig2: higher with model', int((mdl > c).sum()), '| dv range', round(dv.min(), 1), round(dv.max(), 1), '| dd range', round(dd.min(), 3), round(dd.max(), 3),
      '| obs/pch max', round(obs.max(), 1), round(pch.max(), 1), round(obs.min(), 1))

# ---------------- Fig. S2: Bland-Altman ----------------
fig, axs = plt.subplots(1, 2, figsize=(150 * mm, 62 * mm), sharey=True)
for ax, (P, title, mk, col) in zip(axs, [(d[f'{T}_true_vol_pred'] / 1000, 'Conditioned model', 'o', BLUE),
                                         (bl, 'No-change reference', 's', GREY)]):
    mval = (P + fu) / 2; diff = P - fu; b = diff.mean(); sd = diff.std(ddof=1)
    ax.axhline(0, color=GRID, lw=1.0, zorder=0)
    ax.scatter(mval, diff, s=16, marker=mk, facecolor=col, edgecolor='white', linewidth=1.0, zorder=3)
    for yv, lab, ls, va in [(b, f'Bias {b:+.1f}', '-', 'bottom'), (b + 1.96 * sd, f'{b + 1.96 * sd:+.1f}', (0, (3, 2)), 'bottom'),
                            (b - 1.96 * sd, f'{b - 1.96 * sd:+.1f}', (0, (3, 2)), 'top')]:
        ax.axhline(yv, color=INK if ls == '-' else REF, lw=1.2 if ls == '-' else 1.0, ls=ls)
        ax.text(0.98, yv + (1.2 if va == 'bottom' else -1.2), lab.replace('-', '−'), transform=ax.get_yaxis_transform(),
                ha='right', va=va, fontsize=6.2, color=INK if ls == '-' else REF)
    ax.set_title(title, fontsize=7, pad=4); ax.set_xlabel('Mean of predicted and follow-up volume, cm$^3$')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
axs[0].set_ylabel('Predicted − follow-up tumor volume, cm$^3$')
for ax, l in zip(axs, 'ab'): ax.text(-0.12 if l == 'a' else -0.06, 1.04, l, transform=ax.transAxes, fontsize=7, fontweight='bold', va='bottom')
fig.tight_layout(pad=0.5, w_pad=1.2)
for ext in ['pdf', 'eps']: fig.savefig(f'figs/FigS2_srep.{ext}')
fig.savefig('figs/FigS2_srep.png', dpi=600); fig.savefig('figs/FigS2_srep.tif', dpi=1200, pil_kwargs={'compression': 'tiff_lzw'})

# ---------------- Fig. S3: calibration ----------------
cal = np.load(f'data/ucsf/{RUN}/calibration.npz')
fig, ax = plt.subplots(figsize=(80 * mm, 76 * mm))
ax.plot([0, 1], [0, 1], ls=(0, (3, 2)), lw=1.0, color=REF)
for key, lab, col, mk in [('raw', 'Network output', BLUE, 'o'), ('enf', 'After residual-tumor enforcement', ORANGE, '^')]:
    n, p, y = cal[f'{key}_n'], cal[f'{key}_p'], cal[f'{key}_y']; ok = n > 0
    ax.plot(p[ok] / n[ok], y[ok] / n[ok], marker=mk, ms=3, lw=1.0, color=col, label=lab)
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed tumor fraction at follow-up')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); style(ax)
ax.legend(loc='upper left', frameon=False, fontsize=6)
fig.tight_layout(pad=0.4)
for ext in ['pdf', 'eps']: fig.savefig(f'figs/FigS3_srep.{ext}')
fig.savefig('figs/FigS3_srep.png', dpi=600); fig.savefig('figs/FigS3_srep.tif', dpi=1200, pil_kwargs={'compression': 'tiff_lzw'})

# ---------------- Fig. S1: flow ----------------
W, H = 174, 100
fig = plt.figure(figsize=(W * mm, H * mm)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
def box(x, y, w, h, t, fc='white', fs=6.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec='#5F5F5F', lw=1.0))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.25)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>,head_width=0.9,head_length=1.1', mutation_scale=4,
                                 color=INK, lw=1.0, shrinkA=0, shrinkB=0))
ax.text(44, 96, 'Development (single center)', ha='center', fontsize=7.5, fontweight='bold')
ax.text(130, 96, 'External validation (public dataset)', ha='center', fontsize=7.5, fontweight='bold')
DEV, VAL = '#E6EEF8', '#E7F4EE'
box(8, 78, 72, 14, 'Adults with histopathologically confirmed diffuse glioma\nand postoperative MRI surveillance,\nKepler University Hospital Linz (examinations 1999–2021)', fc=DEV)
box(94, 78, 72, 14, 'UCSF-ALPTDG dataset\n298 patients, 596 examinations\n(two consecutive post-treatment time points)', fc=VAL)
arrow(44, 78, 44, 69.5); arrow(130, 78, 130, 69.5)
box(8, 54, 72, 15.5, 'Inclusion: ≥ 2 postoperative MRI examinations;\nhistomorphologic type available. Baseline: first\nexamination ≥ 30 days after surgery; examinations\nduring radiotherapy or chemotherapy excluded')
box(94, 54, 72, 15.5, 'Excluded (n = 186): radiotherapy or chemotherapy\nat either examination (n = 118), treatment\nstatus not documented (n = 1), CNS WHO\ngrade 4 (n = 67)', fs=6.4)
arrow(44, 54, 44, 45.5); arrow(130, 54, 130, 45.5)
box(8, 30, 72, 15.5, 'Development cohort\n159 patients, 206 surgical episodes,\n883 baseline–follow-up pairs (1,089 examinations)', fc=DEV)
box(94, 30, 72, 15.5, 'Validation cohort\n112 patients, 112 baseline–follow-up pairs\n(tumor masks without resection cavity)', fc=VAL)
arrow(26, 30, 26, 21.5); arrow(62, 30, 62, 21.5); arrow(130, 30, 130, 21.5)
box(8, 6, 36, 15.5, 'Training partition\n127 patients\n(703 pairs)')
box(46, 6, 34, 15.5, 'Internal validation\npartition, 32 patients\n(180 pairs; checkpoint\nselection, early stopping)', fs=6.4)
box(94, 6, 72, 15.5, 'Inference with the archived weights of the\nconditioned model; no-change reference;\nlabel controls; analyses of change, time and location')
for ext in ['pdf', 'eps']: fig.savefig(f'figs/FigS1_srep.{ext}')
fig.savefig('figs/FigS1_srep.png', dpi=600); fig.savefig('figs/FigS1_srep.tif', dpi=1200, pil_kwargs={'compression': 'tiff_lzw'})
print('done')
