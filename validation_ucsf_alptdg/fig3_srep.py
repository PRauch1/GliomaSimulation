"""Figure 3 (Scientific Reports version): baseline FLAIR with baseline tumor and resection cavity, follow-up tumor, and
the conditioned model's probability map (primary preprocessing: FLAIR input with N4 bias-field correction, tumor masks).
Usage: python3 fig3_srep.py 100158,100147,100036 (patients selected by fig3_select.py)"""
import os, sys, json, numpy as np, nibabel as nib, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('LGG_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lgg_regrowth')))
from safetensors.torch import load_file
from core.models import ResizeConvUNet3D_FiLM
from eval.eval_io import zscore_normalize, pad_to_multiple, unpad_volume, build_time_channel
import pandas as pd

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'pdf.fonttype': 42})
torch.set_num_threads(2)
cfg = json.load(open(os.path.join(BASE, 'infer/model_config.json')))
m = ResizeConvUNet3D_FiLM(in_channels=3, out_channels=1, channels=tuple(cfg['channels']),
                          cat_vocab_sizes=[len(cfg['vocabs'][k]) for k in cfg['cat_order']], meta_emb_dim=16, meta_dim=64,
                          meta_dropout=0.0)
m.load_state_dict(load_file(os.path.join(BASE, 'infer/model.safetensors'))); m.eval()
fv = pd.read_excel(os.path.join(BASE, 'data/new/feat_val.xlsx')); fv['pid'] = fv.id.str[8:].astype(int); fv = fv.set_index('pid')
days = pd.read_csv(os.path.join(BASE, 'data/nc/per_followup_metrics_0.5_idh_histo_cortex.csv')); days['pid'] = days.patient_id.str[8:].astype(int)
days = days.set_index('pid').fu_day
R = os.path.join(BASE, 'data/ucsf/raw')

def idx(cat, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return 0
    return int(cfg['vocabs'][cat].get(str(val).strip(), 0))

import os
MOD = os.environ.get('MODALITY', 'flair'); N4 = os.environ.get('N4', '1') == '1'
def n4_correct(path):
    import SimpleITK as sitk
    im = sitk.ReadImage(path, sitk.sitkFloat32); mask = sitk.Cast(im > 0, sitk.sitkUInt8)
    sh = sitk.Shrink(im, [4] * 3); ms = sitk.Shrink(mask, [4] * 3)
    c = sitk.N4BiasFieldCorrectionImageFilter(); c.Execute(sh, ms)
    out = im / sitk.Exp(c.GetLogBiasFieldAsImage(im))
    return np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0)).astype(np.float32) * (np.asanyarray(nib.load(path).dataobj) != 0)

def predict(i):
    path = f'{R}/{i}/{i}_time1_{MOD}.nii.gz'
    raw = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
    l1 = np.rint(np.asanyarray(nib.load(f'{R}/{i}/{i}_time1_seg.nii.gz').dataobj)).astype(np.int16)
    l2 = np.rint(np.asanyarray(nib.load(f'{R}/{i}/{i}_time2_seg.nii.gz').dataobj)).astype(np.int16)
    M = ((l1 > 0) & (l1 != 4)).astype(np.float32)
    img = zscore_normalize(n4_correct(path) if N4 else raw); Mf = M
    t = build_time_channel(img, float(days[i]), 4536.0)
    x, pads = pad_to_multiple(np.stack([img, Mf, t], 0).astype(np.float32), 8)
    meta = [idx(c, fv.loc[i, c]) for c in cfg['cat_order']]
    with torch.no_grad(): p = torch.sigmoid(m(torch.from_numpy(x[None]), torch.tensor([meta]))).numpy()
    p = np.maximum(unpad_volume(p, pads), Mf)
    return raw, l1, l2, M.astype(bool), p

ids = [int(x) for x in sys.argv[1].split(',')]
mm = 1 / 25.4
fig, axs = plt.subplots(len(ids), 3, figsize=(174 * mm, 58 * mm * len(ids)))
for r, i in enumerate(ids):
    raw, l1, l2, M, p = predict(i)
    G = (l2 > 0) & (l2 != 4); cav = l1 == 4; P = p >= 0.5
    # slice with the largest union of predicted and observed new tumor (voxels outside the baseline mask)
    new_area = ((P | G) & ~M).sum(axis=(0, 1))
    z = int(np.argmax(new_area))
    sl = lambda v: np.asarray(v[:, :, z]).T
    nz = np.argwhere(sl(raw) > 0); (y0, x0), (y1, x1) = nz.min(0) - 4, nz.max(0) + 4
    crop = lambda a: a[max(y0, 0):y1, max(x0, 0):x1]
    base = crop(sl(raw)); vmax = np.percentile(base[base > 0], 99.5)
    def contour(ax, mask, color, ls='-'):
        mk = crop(sl(mask)).astype(float)
        if mk.any(): ax.contour(mk, levels=[0.5], colors=[color], linewidths=1.0, linestyles=ls)
    for c in range(3):
        ax = axs[r, c] if len(ids) > 1 else axs[c]
        ax.imshow(base, cmap='gray', vmin=0, vmax=vmax, interpolation='nearest'); ax.set_axis_off()
    a0, a1, a2 = (axs[r] if len(ids) > 1 else axs)
    contour(a0, M, '#2ECC40'); contour(a0, cav, '#FFDC00', '--')
    contour(a1, G, '#00E5FF'); contour(a1, cav, '#FFDC00', '--')
    pr = np.ma.masked_less(crop(sl(p)), 0.05)
    im = a2.imshow(pr, cmap='magma', vmin=0, vmax=1, alpha=0.75, interpolation='nearest')
    contour(a2, G, '#00E5FF')
    # scale bar, 20 mm (1-mm isotropic voxels, one pixel per voxel)
    hh, ww = base.shape; xl, yl = a0.get_xlim(), a0.get_ylim()
    a0.plot([ww - 7 - 20, ww - 7], [8, 8], color='white', lw=1.5, solid_capstyle='butt')
    a0.text(ww - 7 - 10, 10.5, '2 cm', color='white', fontsize=7, ha='center', va='top')
    a0.set_xlim(xl); a0.set_ylim(yl)
    if len(ids) > 1:
        a0.text(0.02, 0.98, 'abcdefgh'[r], transform=a0.transAxes, color='white', fontsize=7, fontweight='bold', va='top')
    print(i, 'slice', z, 'new area', int(new_area[z]), 'FP in cavity (cm3)', round(float((P & cav).sum()) / 1000, 1), 'GT', round(float(G.sum()) / 1000, 1),
          'pred', round(float(P.sum()) / 1000, 1), flush=True)
cb = fig.colorbar(im, ax=axs.ravel().tolist() if len(ids) > 1 else axs, fraction=0.012, pad=0.01)
cb.set_label('Predicted probability'); cb.ax.tick_params(labelsize=7, width=1.0); cb.outline.set_linewidth(1.0)
from matplotlib.lines import Line2D
key = [Line2D([], [], color='#2ECC40', lw=1.0, label='Baseline tumor'),
       Line2D([], [], color='#00E5FF', lw=1.0, label='Follow-up tumor'),
       Line2D([], [], color='#FFDC00', lw=1.0, ls='--', label='Resection cavity (baseline)')]
_g0, _g1 = axs[0, 0].get_position(), axs[-1, -1].get_position()
fig.legend(handles=key, loc='upper center', bbox_to_anchor=((_g0.x0 + _g1.x1) / 2, _g1.y0 - 0.012), ncol=3, fontsize=7, frameon=True,
           facecolor='black', edgecolor='black', labelcolor='white', handlelength=2.2, columnspacing=1.6)
fig.savefig(os.path.join(BASE, 'figs/Fig3_srep.png'), dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(BASE, 'figs/Fig3_srep.tif'), dpi=600, bbox_inches='tight', facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
