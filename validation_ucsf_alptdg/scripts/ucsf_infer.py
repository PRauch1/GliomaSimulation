#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU inference of the conditioned model on UCSF-ALPTDG with counterfactual conditioning labels.

Per patient and variant, the script re-runs the earlier evaluation with reconstructed preprocessing (baseline FLAIR or
T2 image as distributed, optional N4, z-score within non-zero voxels, centred padding to a multiple of 8,
residual-mask enforcement, threshold 0.5) and adds:
  - no-change reference (baseline mask vs follow-up mask)
  - metrics restricted to the region of new growth (follow-up voxels outside the baseline mask)
  - counterfactual conditioning (cortex set to unknown, labels permuted across patients, all labels unknown)
  - voxel-level calibration counts (15 bins) within the brain
  - optional tumor-only evaluation (resection cavity removed from residual and ground-truth masks)
Results are appended per patient to <out>/per_patient.csv; calibration counts to <out>/calibration.npz.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, nibabel as nib, torch
from safetensors.torch import load_file

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True, help='path to lgg_regrowth (code)')
p.add_argument('--weights', required=True); p.add_argument('--config', required=True)
p.add_argument('--files', required=True, help='CSV with columns id,t2_bl,seg_bl,seg_fu (absolute paths)')
p.add_argument('--features', required=True, help='features_validation.xlsx')
p.add_argument('--days', required=True, help='per_followup_metrics_0.5_idh_histo_cortex.csv (fu_day used in evaluation)')
p.add_argument('--out', required=True)
p.add_argument('--variants', default='true,cortex_unk,all_perm,all_unk,cortex_perm')
p.add_argument('--tumor-only', action='store_true', help='additional pass with cavity removed from masks')
p.add_argument('--masks', default='lesion', choices=['lesion', 'tumor'], help='mask definition for the counterfactual variants')
p.add_argument('--threads', type=int, default=2)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--flip-axes', default='', help='array axes flipped before inference (reconstructed orientation)')
p.add_argument('--modality', default='flair', choices=['t2', 'flair'], help='baseline image used as model input')
p.add_argument('--n4', action='store_true', help='N4 bias-field correction before z-scoring')
p.add_argument('--extra-lesion', action='store_true', help='additional pass on lesion masks (true labels) for agreement with the earlier evaluation')
p.add_argument('--thresholds', default='0.5,0.6,0.7,0.8,0.9,0.95,0.99')
a = p.parse_args()

sys.path.insert(0, a.repo)
from core.models import ResizeConvUNet3D_FiLM  # noqa: E402
from eval.eval_io import zscore_normalize, pad_to_multiple, unpad_volume, build_time_channel  # noqa: E402
from eval.eval_metrics import dice_score, hd95_mm, nsd_mm  # noqa: E402

torch.set_num_threads(a.threads)

def n4_correct(path):
    import SimpleITK as sitk
    im = sitk.ReadImage(path, sitk.sitkFloat32); mask = sitk.Cast(im > 0, sitk.sitkUInt8)
    sh = sitk.Shrink(im, [4] * 3); ms = sitk.Shrink(mask, [4] * 3)
    c = sitk.N4BiasFieldCorrectionImageFilter(); c.Execute(sh, ms)
    out = im / sitk.Exp(c.GetLogBiasFieldAsImage(im))
    arr = np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0)).astype(np.float32)
    return arr * (np.asanyarray(nib.load(path).dataobj) != 0)
THR = [float(x) for x in a.thresholds.split(',')]
cfg = json.load(open(a.config))
model = ResizeConvUNet3D_FiLM(in_channels=3, out_channels=1, channels=tuple(cfg['channels']),
                              cat_vocab_sizes=[len(cfg['vocabs'][k]) for k in cfg['cat_order']],
                              meta_emb_dim=cfg['meta_emb_dim'], meta_dim=cfg['meta_dim'], meta_dropout=cfg['meta_dropout'])
model.load_state_dict(load_file(a.weights), strict=True); model.eval()
T_MAX = json.load(open(os.path.join(os.path.dirname(a.config), 'preprocessing.json')))['time_encoding']['t_max_days']

files = pd.read_csv(a.files).set_index('id')
fv = pd.read_excel(a.features); fv['pid'] = fv.id.str.replace('Patient_', '').astype(int); fv = fv.set_index('pid')
days = pd.read_csv(a.days); days['pid'] = days.patient_id.str.replace('Patient_', '').astype(int)
days = days.set_index('pid').fu_day
ids = [i for i in files.index if i in fv.index and i in days.index]
if a.limit: ids = ids[:a.limit]

def idx(cat, val):
    voc = cfg['vocabs'][cat]
    if val is None or (isinstance(val, float) and np.isnan(val)): return 0
    return int(voc.get(str(val).strip(), 0))

true_meta = {i: [idx(c, fv.loc[i, c]) for c in cfg['cat_order']] for i in ids}
rng = np.random.default_rng(42)
perm_all = dict(zip(ids, [true_meta[j] for j in rng.permutation(ids)]))
perm_cortex = dict(zip(ids, [true_meta[j][0] for j in np.random.default_rng(7).permutation(ids)]))

def meta_for(i, v):
    m = list(true_meta[i])
    if v == 'true': return m
    if v == 'cortex_unk': return [0] + m[1:]
    if v == 'cortex_perm': return [perm_cortex[i]] + m[1:]
    if v == 'all_perm': return list(perm_all[i])
    if v == 'all_unk': return [0, 0, 0]
    if v == 'cortex_swap':  # every patient receives a different, in-distribution cortex category
        iso, meso = cfg['vocabs']['cortex']['Isocortex'], cfg['vocabs']['cortex']['Mesocortex']
        return [meso if m[0] == iso else iso] + m[1:]
    raise ValueError(v)

@torch.no_grad()
def predict(img, res, day, meta):
    t = build_time_channel(img, day, T_MAX)
    x, pads = pad_to_multiple(np.stack([img, res, t], 0).astype(np.float32), 8)
    prob = torch.sigmoid(model(torch.from_numpy(x[None]), torch.tensor([meta]).long())).numpy()
    return unpad_volume(prob, pads).astype(np.float32)

def metrics(pred, gt, hdr, pre=''):
    out = {f'{pre}dice': dice_score(pred, gt), f'{pre}hd95': hd95_mm(pred, gt, hdr),
           f'{pre}nsd1': nsd_mm(pred, gt, hdr, 1.0), f'{pre}nsd2': nsd_mm(pred, gt, hdr, 2.0)}
    vv = float(np.prod(hdr.get_zooms()[:3]))
    out[f'{pre}vol_pred'] = float(pred.sum()) * vv; out[f'{pre}vol_gt'] = float(gt.sum()) * vv
    return out

def growth_metrics(pred, M, G, vv, pre=''):
    N = G & ~M; A = pred & ~M; inter = (A & N).sum()
    return {f'{pre}new_gt': float(N.sum()) * vv, f'{pre}new_pred': float(A.sum()) * vv, f'{pre}new_tp': float(inter) * vv,
            f'{pre}new_dice': float(2 * inter / (A.sum() + N.sum())) if (A.sum() + N.sum()) else 1.0,
            f'{pre}regressed': float((M & ~G).sum()) * vv}

os.makedirs(a.out, exist_ok=True)
out_csv = os.path.join(a.out, 'per_patient.csv')
done = set(pd.read_csv(out_csv).id) if os.path.exists(out_csv) else set()
calib_path = os.path.join(a.out, 'calibration.npz')
NB = 15
cal = dict(np.load(calib_path)) if os.path.exists(calib_path) else {k: np.zeros(NB) for k in
      ['raw_n', 'raw_p', 'raw_y', 'enf_n', 'enf_p', 'enf_y']}
variants = a.variants.split(',')
for n, i in enumerate(ids):
    if i in done: continue
    t0 = time.time()
    f = files.loc[i]
    img_path = f.t2_bl.replace('_time1_t2.nii.gz', '_time1_flair.nii.gz') if a.modality == 'flair' else f.t2_bl
    im = nib.load(img_path); s1 = nib.load(f.seg_bl); s2 = nib.load(f.seg_fu)
    FA = tuple(int(c) for c in a.flip_axes if c.isdigit())
    flip = (lambda v: np.ascontiguousarray(np.flip(v, FA))) if FA else (lambda v: v)
    raw = flip(np.asanyarray(im.dataobj).astype(np.float32))
    img = zscore_normalize(flip(n4_correct(img_path)) if a.n4 else raw)
    l1 = flip(np.rint(np.asanyarray(s1.dataobj)).astype(np.int16)); l2 = flip(np.rint(np.asanyarray(s2.dataobj)).astype(np.int16))
    assert img.shape == l1.shape == l2.shape, (i, img.shape, l1.shape, l2.shape)
    hdr = s2.header; vv = float(np.prod(hdr.get_zooms()[:3])); brain = raw != 0
    row = {'id': i, 'day': float(days[i]), 'meta_true': ','.join(map(str, true_meta[i]))}
    lesion_def = ('lesion', l1 > 0, l2 > 0); tumor_def = ('tumor', (l1 > 0) & (l1 != 4), (l2 > 0) & (l2 != 4))
    defs = ([tumor_def] + ([lesion_def] if a.extra_lesion else [])) if a.masks == 'tumor' else [lesion_def] + ([tumor_def] if a.tumor_only else [])
    main = defs[0][0]
    for name, M, G in defs:
        row.update(metrics(M, G, hdr, f'{name}_cf_'))
        row.update(growth_metrics(M, M, G, vv, f'{name}_cf_'))
        vlist = variants if name == main else ['true']
        p_true = None
        for v in vlist:
            pr = predict(img, M.astype(np.float32), float(days[i]), meta_for(i, v))
            pe = np.maximum(pr, M.astype(np.float32))
            P = pe >= 0.5; Praw = pr >= 0.5
            pre = f'{name}_{v}_'
            row.update(metrics(P, G, hdr, pre)); row[pre + 'dice_raw'] = dice_score(Praw, G)
            row[pre + 'vol_raw'] = float(Praw.sum()) * vv; row[pre + 'soft_vol_raw'] = float(pr.sum()) * vv
            row.update(growth_metrics(P, M, G, vv, pre))
            cav1 = l1 == 4; cav2 = l2 == 4
            row[pre + 'pred_in_bl_cavity'] = float((P & cav1).sum()) * vv; row[pre + 'pred_in_fu_cavity'] = float((P & cav2).sum()) * vv
            row[pre + 'fp'] = float((P & ~G).sum()) * vv; row[pre + 'fn'] = float((~P & G).sum()) * vv
            row[pre + 'fp_in_bl_cavity'] = float((P & cav1 & ~G).sum()) * vv
            if v == 'true' and name == main:
                Pn = P & ~cav1
                row[pre + 'nocav_dice'] = dice_score(Pn, G); row[pre + 'nocav_vol'] = float(Pn.sum()) * vv
            if v == 'true':
                p_true = pr
                if name == main:
                    for th in THR:
                        Pt = pe >= th; At = Pt & ~M; Nt = G & ~M
                        row[f'{pre}thr{th:g}_dice'] = dice_score(Pt, G); row[f'{pre}thr{th:g}_vol'] = float(Pt.sum()) * vv
                        row[f'{pre}thr{th:g}_new_pred'] = float(At.sum()) * vv; row[f'{pre}thr{th:g}_new_tp'] = float((At & Nt).sum()) * vv
                if name == main:
                    for key, pp in [('raw', pr), ('enf', pe)]:
                        b = np.minimum((pp[brain] * NB).astype(int), NB - 1); y = G[brain].astype(float)
                        cal[f'{key}_n'] += np.bincount(b, minlength=NB); cal[f'{key}_p'] += np.bincount(b, pp[brain], NB)
                        cal[f'{key}_y'] += np.bincount(b, y, NB)
            else:
                row[pre + 'mean_abs_prob_diff'] = float(np.abs(pr - p_true)[brain].mean())
                row[pre + 'changed_voxels'] = float(((pr >= 0.5) != (p_true >= 0.5)).sum()) * vv
    row['sec'] = round(time.time() - t0, 1)
    pd.DataFrame([row]).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
    np.savez(calib_path, **cal)
    print(f'[{n + 1}/{len(ids)}] {i}: Dice true {row.get(main + "_true_dice", float("nan")):.4f} '
          f'cf {row[main + "_cf_dice"]:.4f} ({row["sec"]} s)', flush=True)
print('done')
