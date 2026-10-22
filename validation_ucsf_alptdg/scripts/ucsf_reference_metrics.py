#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-change (carry-forward) reference on UCSF-ALPTDG segmentations.

For each validation patient, the baseline segmentation is compared with the follow-up segmentation, both in the
co-registered space distributed with the dataset. Metric definitions are copied from
lgg_regrowth/eval/eval_metrics.py (Dice; HD95 and NSD from surface voxels obtained by 3x3x3 erosion).

Two mask definitions:
  lesion = all labels > 0 (NETC 1, SNFH 2, ET 3, resection cavity 4) -- definition used in the original evaluation
  tumor  = labels 1-3 (resection cavity excluded)

Usage:
  python ucsf_reference_metrics.py --seg-root <folder with UCSF segmentations> --cohort dataset_validation.csv \
      --out no_change_reference.csv [--pattern "{id}_time{t}*seg*.nii*"]
"""
import argparse, fnmatch, glob, os, re, sys
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


# ---------------- metrics (identical to lgg_regrowth/eval/eval_metrics.py) ----------------
def dice_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum(); denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / (denom + eps))


def _surface_points(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    er = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.argwhere(mask ^ er).astype(np.float32)


def _spacing_zyx_mm(header):
    sx, sy, sz = header.get_zooms()[:3]
    return np.array([sz, sy, sx], dtype=np.float32)


def _surface_dists(pred, gt, header):
    A = _surface_points(pred) * _spacing_zyx_mm(header); B = _surface_points(gt) * _spacing_zyx_mm(header)
    dA, _ = cKDTree(B).query(A, k=1); dB, _ = cKDTree(A).query(B, k=1)
    return dA, dB


def hd95_and_nsd(pred, gt, header, tols=(1.0, 2.0)):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, {t: 1.0 for t in tols}
    if pred.sum() == 0 or gt.sum() == 0:
        return 999.0, {t: 0.0 for t in tols}
    dA, dB = _surface_dists(pred, gt, header)
    hd95 = float(np.percentile(np.concatenate([dA, dB]), 95))
    nsd = {t: float(((dA <= t).sum() + (dB <= t).sum()) / max(1, len(dA) + len(dB))) for t in tols}
    return hd95, nsd


# ---------------- file discovery ----------------
DEFAULT_PATTERNS = ["*{id}*time{t}*seg*.nii*", "*{id}*time_{t}*seg*.nii*", "*{id}*_{t}_*seg*.nii*",
                    "*{id}*tp{t}*seg*.nii*", "*{id}*time{t}*label*.nii*", "*{id}*time{t}*mask*.nii*"]


def find_seg(all_files, pid, t, patterns):
    for p in patterns:
        pat = p.format(id=pid, t=t).lower()
        hits = [f for f in all_files if fnmatch.fnmatch(os.path.basename(f).lower(), pat)
                or fnmatch.fnmatch(f.lower(), '*' + pat)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RuntimeError(f'{pid} time {t}: several candidate files for pattern {p}: {hits[:5]}')
    return None


def boot_ci(x, n=10000, seed=42):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg-root', required=True)
    ap.add_argument('--cohort', required=True, help='dataset_validation.csv (semicolon-separated)')
    ap.add_argument('--out', default='no_change_reference.csv')
    ap.add_argument('--pattern', action='append', default=None, help='filename pattern with {id} and {t}')
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort, sep=';')
    ids = sorted(cohort.id.dropna().astype(int).unique())
    vol_csv = cohort.dropna(subset=['id']).assign(id=lambda d: d.id.astype(int), timepoint=lambda d: d.timepoint.astype(int))
    vol_csv = vol_csv.set_index(['id', 'timepoint'])['volume']
    all_files = [f for f in glob.glob(os.path.join(args.seg_root, '**', '*.nii*'), recursive=True)]
    patterns = args.pattern or DEFAULT_PATTERNS
    print(f'{len(ids)} patients, {len(all_files)} NIfTI files under {args.seg_root}')

    rows, missing = [], []
    for pid in ids:
        f1, f2 = find_seg(all_files, pid, 1, patterns), find_seg(all_files, pid, 2, patterns)
        if f1 is None or f2 is None:
            missing.append(pid); continue
        i1, i2 = nib.load(f1), nib.load(f2)
        s1 = np.asanyarray(i1.dataobj).astype(np.int16); s2 = np.asanyarray(i2.dataobj).astype(np.int16)
        if s1.shape != s2.shape or not np.allclose(i1.affine, i2.affine, atol=1e-3):
            raise RuntimeError(f'{pid}: baseline and follow-up segmentations are not in the same space')
        vv = float(np.prod(i2.header.get_zooms()[:3]))
        r = {'id': pid, 'file_bl': os.path.relpath(f1, args.seg_root), 'file_fu': os.path.relpath(f2, args.seg_root),
             'labels_bl': ','.join(map(str, np.unique(s1)[1:])), 'labels_fu': ','.join(map(str, np.unique(s2)[1:])),
             'voxel_mm3': vv}
        for name, fn in [('lesion', lambda s: s > 0), ('tumor', lambda s: (s > 0) & (s != 4))]:
            M, G = fn(s1), fn(s2)
            hd, nsd = hd95_and_nsd(M, G, i2.header)
            r.update({
                f'{name}_vol_bl_mm3': M.sum() * vv, f'{name}_vol_fu_mm3': G.sum() * vv,
                f'{name}_dice': dice_score(M, G), f'{name}_hd95_mm': hd,
                f'{name}_nsd1': nsd[1.0], f'{name}_nsd2': nsd[2.0],
                f'{name}_abs_vol_err_mm3': abs(M.sum() - G.sum()) * vv,
                f'{name}_new_mm3': np.logical_and(G, ~M).sum() * vv,       # follow-up voxels outside baseline
                f'{name}_regressed_mm3': np.logical_and(M, ~G).sum() * vv,  # baseline voxels absent at follow-up
            })
        r['check_vol_bl'] = abs(r['lesion_vol_bl_mm3'] - vol_csv.get((pid, 1), np.nan)) < 0.5 * vv
        r['check_vol_fu'] = abs(r['lesion_vol_fu_mm3'] - vol_csv.get((pid, 2), np.nan)) < 0.5 * vv
        rows.append(r)
        print(f"{pid}: lesion Dice {r['lesion_dice']:.3f} HD95 {r['lesion_hd95_mm']:.2f} | tumor Dice {r['tumor_dice']:.3f}"
              f" | volume check {r['check_vol_bl']}/{r['check_vol_fu']}")

    if missing:
        print(f'WARNING: no segmentation pair found for {len(missing)} patients: {missing[:20]}', file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'\nwritten {args.out} ({len(df)} patients)')
    print(f"volume check against dataset table: baseline {df.check_vol_bl.sum()}/{len(df)}, follow-up {df.check_vol_fu.sum()}/{len(df)}")
    for name in ['lesion', 'tumor']:
        print(f'\n== {name} ==')
        for m in ['dice', 'hd95_mm', 'nsd1', 'nsd2']:
            x = df[f'{name}_{m}']
            lo, hi = boot_ci(x)
            print(f'{m:8s} mean {x.mean():.4f} (95% CI {lo:.4f}-{hi:.4f}) SD {x.std(ddof=1):.4f} median {x.median():.4f}')
        ae = df[f'{name}_abs_vol_err_mm3'] / 1000
        print(f'abs vol err cm3 mean {ae.mean():.2f} median {ae.median():.2f}')
        print(f"new-growth volume cm3 median {df[f'{name}_new_mm3'].median()/1000:.2f}; "
              f"regressed volume cm3 median {df[f'{name}_regressed_mm3'].median()/1000:.2f}")


if __name__ == '__main__':
    main()
