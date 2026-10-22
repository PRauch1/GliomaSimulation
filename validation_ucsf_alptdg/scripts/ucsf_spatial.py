#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Magnitude-independent spatial evaluation of the conditioned model on UCSF-ALPTDG (tumor masks, FLAIR + N4).

Analyses specified after the primary analysis (per patient):
  1. Voxel-level ranking of new growth within a peritumoral shell (brain voxels outside the baseline tumor mask within
     SHELL mm of it): AUC of the model probability (correct labels, cortical architecture swapped, labels permuted)
     versus two model-free rankings (proximity = negative distance to the baseline tumor; FLAIR intensity).
  2. Oracle-volume placement: each ranking adds exactly the observed new-growth volume (top-k shell voxels);
     Dice of new voxels and of the full follow-up mask (baseline + added voxels).
  3. Volume-matched placement: proximity ranking adds the model's own predicted new volume (threshold 0.5).
  4. Time diagnostic: predicted new volume with the interval set to zero (correct labels).
Results are appended per patient to <out>/per_patient.csv.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, nibabel as nib, torch
from safetensors.torch import load_file
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--weights', required=True); p.add_argument('--config', required=True)
p.add_argument('--files', required=True); p.add_argument('--features', required=True); p.add_argument('--days', required=True)
p.add_argument('--out', required=True); p.add_argument('--shell', type=float, default=15.0); p.add_argument('--threads', type=int, default=2)
a = p.parse_args()
sys.path.insert(0, a.repo)
from core.models import ResizeConvUNet3D_FiLM  # noqa: E402
from eval.eval_io import zscore_normalize, pad_to_multiple, unpad_volume, build_time_channel  # noqa: E402

torch.set_num_threads(a.threads)
cfg = json.load(open(a.config))
model = ResizeConvUNet3D_FiLM(in_channels=3, out_channels=1, channels=tuple(cfg['channels']),
                              cat_vocab_sizes=[len(cfg['vocabs'][k]) for k in cfg['cat_order']],
                              meta_emb_dim=cfg['meta_emb_dim'], meta_dim=cfg['meta_dim'], meta_dropout=cfg['meta_dropout'])
model.load_state_dict(load_file(a.weights), strict=True); model.eval()
T_MAX = json.load(open(os.path.join(os.path.dirname(a.config), 'preprocessing.json')))['time_encoding']['t_max_days']

files = pd.read_csv(a.files).set_index('id')
fv = pd.read_excel(a.features); fv['pid'] = fv.id.str.replace('Patient_', '').astype(int); fv = fv.set_index('pid')
days = pd.read_csv(a.days); days['pid'] = days.patient_id.str.replace('Patient_', '').astype(int); days = days.set_index('pid').fu_day
ids = [i for i in files.index if i in fv.index and i in days.index]

def idx(cat, val):
    voc = cfg['vocabs'][cat]
    if val is None or (isinstance(val, float) and np.isnan(val)): return 0
    return int(voc.get(str(val).strip(), 0))
true_meta = {i: [idx(c, fv.loc[i, c]) for c in cfg['cat_order']] for i in ids}
rng = np.random.default_rng(42)                      # identical permutation to the primary run
perm_all = dict(zip(ids, [true_meta[j] for j in rng.permutation(ids)]))
ISO, MESO = cfg['vocabs']['cortex']['Isocortex'], cfg['vocabs']['cortex']['Mesocortex']
def meta_for(i, v):
    m = list(true_meta[i])
    if v == 'true': return m
    if v == 'perm': return list(perm_all[i])
    if v == 'swap': return [MESO if m[0] == ISO else ISO] + m[1:]
    raise ValueError(v)

def n4_correct(path):
    import SimpleITK as sitk
    im = sitk.ReadImage(path, sitk.sitkFloat32); mask = sitk.Cast(im > 0, sitk.sitkUInt8)
    sh = sitk.Shrink(im, [4] * 3); ms = sitk.Shrink(mask, [4] * 3)
    c = sitk.N4BiasFieldCorrectionImageFilter(); c.Execute(sh, ms)
    out = im / sitk.Exp(c.GetLogBiasFieldAsImage(im))
    return np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0)).astype(np.float32) * (np.asanyarray(nib.load(path).dataobj) != 0)

@torch.no_grad()
def predict(img, res, day, meta):
    t = build_time_channel(img, day, T_MAX)
    x, pads = pad_to_multiple(np.stack([img, res, t], 0).astype(np.float32), 8)
    prob = torch.sigmoid(model(torch.from_numpy(x[None]), torch.tensor([meta]).long())).numpy()
    return unpad_volume(prob, pads).astype(np.float32)

def topk_mask(score, sel, k, seed):
    """Boolean mask of the k highest-scoring voxels among sel (ties broken at random, fixed seed)."""
    out = np.zeros(sel.shape, bool)
    if k <= 0: return out
    idxs = np.flatnonzero(sel); s = score.ravel()[idxs]
    jit = np.random.default_rng(seed).random(len(s)) * 1e-6 * (np.std(s) + 1e-12)
    order = np.argsort(-(s + jit), kind='stable')[:min(k, len(idxs))]
    out.ravel()[idxs[order]] = True
    return out

def dice(A, B):
    s = A.sum() + B.sum()
    return float(2 * (A & B).sum() / s) if s else 1.0

os.makedirs(a.out, exist_ok=True)
out_csv = os.path.join(a.out, 'per_patient.csv')
done = set(pd.read_csv(out_csv).id) if os.path.exists(out_csv) else set()
for n, i in enumerate(ids):
    if i in done: continue
    t0 = time.time(); f = files.loc[i]
    path = f.t2_bl.replace('_time1_t2.nii.gz', '_time1_flair.nii.gz')
    raw = np.asanyarray(nib.load(path).dataobj).astype(np.float32); brain = raw != 0
    img = zscore_normalize(n4_correct(path))
    l1 = np.rint(np.asanyarray(nib.load(f.seg_bl).dataobj)).astype(np.int16)
    l2 = np.rint(np.asanyarray(nib.load(f.seg_fu).dataobj)).astype(np.int16)
    M = (l1 > 0) & (l1 != 4); G = (l2 > 0) & (l2 != 4); N = G & ~M
    dist = distance_transform_edt(~M)
    shell = brain & ~M & (dist <= a.shell)
    row = {'id': i, 'day': float(days[i]), 'new_total': int(N.sum()), 'new_in_shell': int((N & shell).sum()),
           'shell_vox': int(shell.sum()), 'dice_nochange': dice(M, G)}
    scores = {'proximity': -dist, 'intensity': img}
    for v in ['true', 'swap', 'perm']:
        scores[f'model_{v}'] = predict(img, M.astype(np.float32), float(days[i]), meta_for(i, v))
    p0 = predict(img, M.astype(np.float32), 0.0, meta_for(i, 'true'))
    row['pred_new_t'] = int(((scores['model_true'] >= 0.5) & ~M).sum()); row['pred_new_t0'] = int(((p0 >= 0.5) & ~M).sum())
    row['soft_new_t'] = float(scores['model_true'][~M & brain].sum()); row['soft_new_t0'] = float(p0[~M & brain].sum())
    y = N[shell]
    for k, s in scores.items():
        row[f'auc_{k}'] = float(roc_auc_score(y, s[shell])) if 0 < y.sum() < y.size else np.nan
        A = topk_mask(s, shell, int(N.sum()), seed=i)                       # oracle volume
        row[f'oracle_newdice_{k}'] = dice(A, N) if N.sum() else np.nan
        row[f'oracle_dice_{k}'] = dice(M | A, G)
    # volume-matched: proximity ranking with the model's own predicted new volume
    Am = topk_mask(scores['proximity'], shell, row['pred_new_t'], seed=i)
    Pm = (scores['model_true'] >= 0.5) & ~M
    row['matched_newdice_model'] = dice(Pm, N) if (Pm.sum() + N.sum()) else np.nan
    row['matched_newdice_proximity'] = dice(Am, N) if (Am.sum() + N.sum()) else np.nan
    row['matched_dice_model'] = dice(M | Pm, G); row['matched_dice_proximity'] = dice(M | Am, G)
    row['matched_prec_model'] = float((Pm & N).sum() / Pm.sum()) if Pm.sum() else np.nan
    row['matched_prec_proximity'] = float((Am & N).sum() / Am.sum()) if Am.sum() else np.nan
    row['sec'] = round(time.time() - t0, 1)
    pd.DataFrame([row]).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
    print(f"[{n + 1}/{len(ids)}] {i}: AUC model {row['auc_model_true']:.3f} prox {row['auc_proximity']:.3f} "
          f"| new t {row['pred_new_t']} t0 {row['pred_new_t0']} ({row['sec']} s)", flush=True)
print('done')
