# -*- coding: utf-8 -*-
"""Tumor-only evaluation (resection cavity excluded): conditioned model (re-run) vs no-change reference,
and conditioning-label controls. Input: data/ucsf/infer_tumor/per_patient.csv, calibration.npz."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

B = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get('UCSF_RUN', 'infer_flair'); OUT = os.environ.get('UCSF_OUT', 'tumor_results.json')
d = pd.read_csv(os.path.join(B, 'data/ucsf', RUN, 'per_patient.csv')).drop_duplicates('id').set_index('id')
n = len(d)

def boot(x, n_b=10000, f=np.mean):
    x = np.asarray(x, float); r = np.random.default_rng(42).integers(0, len(x), (n_b, len(x)))
    return [float(v) for v in np.percentile(f(x[r], axis=1), [2.5, 97.5])]

def summ(x):
    x = pd.Series(x).astype(float)
    return {'mean': float(x.mean()), 'sd': float(x.std(ddof=1)), 'median': float(x.median()),
            'q1': float(x.quantile(.25)), 'q3': float(x.quantile(.75)), 'ci': boot(x)}

def paired(model, ref, higher_better=True):
    dd = (model - ref).astype(float)
    w = stats.wilcoxon(model, ref) if (dd != 0).any() else None
    return {'diff': float(dd.mean()), 'ci': boot(dd), 'median_diff': float(dd.median()), 'p': float(w.pvalue) if w else 1.0,
            'n_model_better': int(((dd > 0) if higher_better else (dd < 0)).sum()), 'n_equal': int((dd == 0).sum())}

R = {'n': n}
T = 'tumor'
cf = {m: d[f'{T}_cf_{m}'] for m in ['dice', 'hd95', 'nsd1', 'nsd2']}
cf['ave'] = (d[f'{T}_cf_vol_pred'] - d[f'{T}_cf_vol_gt']).abs() / 1000
R['cf'] = {k: summ(v) for k, v in cf.items()}
for v in ['true', 'all_perm', 'cortex_swap']:
    if f'{T}_{v}_dice' not in d: continue
    m = {k: d[f'{T}_{v}_{k}'] for k in ['dice', 'hd95', 'nsd1', 'nsd2']}
    m['ave'] = (d[f'{T}_{v}_vol_pred'] - d[f'{T}_{v}_vol_gt']).abs() / 1000
    m['bias'] = (d[f'{T}_{v}_vol_pred'] - d[f'{T}_{v}_vol_gt']) / 1000
    m['dice_raw'] = d[f'{T}_{v}_dice_raw']
    R[v] = {k: summ(x) for k, x in m.items()}
    R[v]['bias_loa'] = [float(m['bias'].mean() - 1.96 * m['bias'].std(ddof=1)), float(m['bias'].mean() + 1.96 * m['bias'].std(ddof=1))]
    R[v]['r'] = float(np.corrcoef(d[f'{T}_{v}_vol_pred'], d[f'{T}_{v}_vol_gt'])[0, 1])
    R[v]['vs_cf'] = {k: paired(m[k], cf[k], higher_better=(k in ('dice', 'nsd1', 'nsd2'))) for k in ['dice', 'hd95', 'nsd1', 'nsd2', 'ave']}
    # new growth
    R[v]['new_dice'] = summ(d[f'{T}_{v}_new_dice'])
    tp = d[f'{T}_{v}_new_tp']; pr = d[f'{T}_{v}_new_pred']; gt = d[f'{T}_{v}_new_gt']
    R[v]['new_precision_pooled'] = float(tp.sum() / pr.sum()); R[v]['new_recall_pooled'] = float(tp.sum() / gt.sum())
    R[v]['new_pred_cm3'] = summ(pr / 1000)
    # predicted vs observed change
    bl = d[f'{T}_cf_vol_pred']; fu = d[f'{T}_cf_vol_gt']
    obs = (fu - bl) / 1000; pch = (d[f'{T}_{v}_vol_pred'] - bl) / 1000
    grow = obs > 0
    R[v]['pred_change'] = summ(pch)
    R[v]['spearman_abs'] = [float(x) for x in stats.spearmanr(pch, obs)]
    R[v]['spearman_rel'] = [float(x) for x in stats.spearmanr(pch / bl, obs / bl)]
    R[v]['auc_abs'] = float(roc_auc_score(grow, pch)); R[v]['auc_rel'] = float(roc_auc_score(grow, pch / bl))
    rr = np.random.default_rng(42); aa, ar = [], []
    for _ in range(2000):
        ix = rr.integers(0, n, n); y = grow.values[ix]
        if y.all() or (~y).all(): continue
        aa.append(roc_auc_score(y, pch.values[ix])); ar.append(roc_auc_score(y, (pch / bl).values[ix]))
    R[v]['auc_abs_ci'] = [float(np.percentile(aa, 2.5)), float(np.percentile(aa, 97.5))]
    R[v]['auc_rel_ci'] = [float(np.percentile(ar, 2.5)), float(np.percentile(ar, 97.5))]
R['obs_change_cm3'] = summ((d[f'{T}_cf_vol_gt'] - d[f'{T}_cf_vol_pred']) / 1000)
R['n_decrease'] = int((d[f'{T}_cf_vol_gt'] < d[f'{T}_cf_vol_pred']).sum())
# label controls: true vs alternatives
R['labels'] = {}
for v in ['all_perm', 'cortex_swap']:
    if f'{T}_{v}_dice' not in d: continue
    R['labels'][v] = {k: paired(d[f'{T}_true_{k}'], d[f'{T}_{v}_{k}'], higher_better=(k != 'hd95')) for k in ['dice', 'hd95', 'nsd1']}
    ae_t = (d[f'{T}_true_vol_pred'] - d[f'{T}_true_vol_gt']).abs() / 1000; ae_v = (d[f'{T}_{v}_vol_pred'] - d[f'{T}_{v}_vol_gt']).abs() / 1000
    R['labels'][v]['ave'] = paired(ae_t, ae_v, higher_better=False)
    R['labels'][v]['mean_abs_prob_diff'] = summ(d[f'{T}_{v}_mean_abs_prob_diff'])
    R['labels'][v]['changed_voxels_cm3'] = summ(d[f'{T}_{v}_changed_voxels'] / 1000)
    R['labels'][v]['n_identical_meta'] = None
# calibration
cal_p = os.path.join(B, 'data/ucsf', RUN, 'calibration.npz')
if os.path.exists(cal_p):
    c = np.load(cal_p)
    for k in ['raw', 'enf']:
        nn, pp, yy = c[f'{k}_n'], c[f'{k}_p'], c[f'{k}_y']; ok = nn > 0
        ece = float(np.sum(np.abs(pp[ok] / nn[ok] - yy[ok] / nn[ok]) * nn[ok]) / nn.sum())
        R[f'calibration_{k}'] = {'ece': ece, 'bins': [[float(pp[i] / nn[i]), float(yy[i] / nn[i]), int(nn[i])] for i in range(len(nn)) if nn[i] > 0]}
# threshold sweep (true labels) and cavity analyses, if available
thr = sorted({c.split('_thr')[1].split('_')[0] for c in d.columns if c.startswith(f'{T}_true_thr')}, key=float)
if thr:
    R['threshold_sweep'] = {}
    for t in thr:
        dt = d[f'{T}_true_thr{t}_dice']; pt = paired(dt, cf['dice'])
        tp = d[f'{T}_true_thr{t}_new_tp']; pr = d[f'{T}_true_thr{t}_new_pred']; gt = d[f'{T}_true_new_gt']
        R['threshold_sweep'][t] = {'dice': summ(dt), 'vs_cf': pt, 'new_pred_cm3_median': float((pr / 1000).median()),
                                   'new_precision_pooled': float(tp.sum() / pr.sum()) if pr.sum() else None,
                                   'new_recall_pooled': float(tp.sum() / gt.sum())}
if f'{T}_true_fp_in_bl_cavity' in d:
    fp = d[f'{T}_true_fp']; fpc = d[f'{T}_true_fp_in_bl_cavity']; hc = d[f'{T}_cf_vol_pred'] >= 0  # all
    cav_any = d.get('cav_bl_vol') if 'cav_bl_vol' in d else None
    R['cavity'] = {'fp_cm3': summ(fp / 1000), 'fp_in_cav_cm3': summ(fpc / 1000),
                   'pooled_share': float(fpc.sum() / fp.sum()),
                   'share_per_patient': summ((fpc / fp.replace(0, np.nan)).fillna(0)),
                   'n_share_gt_half': int(((fpc / fp.replace(0, np.nan)).fillna(0) > 0.5).sum())}
    if f'{T}_true_nocav_dice' in d:
        R['cavity']['nocav'] = {'dice': summ(d[f'{T}_true_nocav_dice']), 'vs_cf': paired(d[f'{T}_true_nocav_dice'], cf['dice']),
                                'vs_true': paired(d[f'{T}_true_nocav_dice'], d[f'{T}_true_dice'])}
# agreement with the earlier evaluation (lesion masks, true labels)
if 'lesion_true_dice' in d:
    pub = pd.read_csv(os.path.join(B, 'data/nc/per_followup_metrics_0.5_idh_histo_cortex.csv'))
    pub['id'] = pub.patient_id.str[8:].astype(int); pub = pub.set_index('id').loc[d.index]
    diff = d['lesion_true_dice'] - pub['dice_enf']
    vr = d['lesion_true_vol_pred'] / pub['vol_pred_enf_mm3']
    gr, go = (d['lesion_true_vol_pred'] - d['lesion_cf_vol_pred']) / 1000, (pub['vol_pred_enf_mm3'] - d['lesion_cf_vol_pred']) / 1000
    gt_match = np.isclose(d['lesion_true_vol_gt'], pub['vol_gt_mm3'])
    gt_maxdiff = float((d['lesion_true_vol_gt'] - pub['vol_gt_mm3']).abs().max())
    R['agreement'] = {'rerun_dice': summ(d['lesion_true_dice']), 'earlier_dice': summ(pub['dice_enf']), 'diff': summ(diff),
                      'abs_diff': summ(diff.abs()), 'n_absdiff_gt_005': int((diff.abs() > 0.05).sum()),
                      'pearson_r': float(np.corrcoef(d['lesion_true_dice'], pub['dice_enf'])[0, 1]),
                      'icc_like_spearman': float(stats.spearmanr(d['lesion_true_dice'], pub['dice_enf'])[0]),
                      'vol_ratio': summ(vr), 'growth_rerun_cm3': summ(gr), 'growth_earlier_cm3': summ(go),
                      'lesion_cf_dice': summ(d['lesion_cf_dice']),
                      'earlier_vs_cf': paired(pub['dice_enf'], d['lesion_cf_dice']),
                      'rerun_vs_cf': paired(d['lesion_true_dice'], d['lesion_cf_dice']),
                      'gt_identical_n': int(gt_match.sum()), 'gt_max_abs_diff_voxels': gt_maxdiff}
json.dump(R, open(os.path.join(B, 'data/ucsf', OUT), 'w'), indent=1)
if __name__ == '__main__':
    print('n =', n)
    for v in ['true', 'all_perm', 'cortex_swap']:
        if v in R:
            print(v, {k: round(R[v][k]['mean'], 4) for k in ['dice', 'hd95', 'nsd1', 'nsd2', 'ave']},
                  'vs cf dice', {k: (round(x, 4) if isinstance(x, float) else x) for k, x in R[v]['vs_cf']['dice'].items() if k != 'ci'})
    print('cf', {k: round(R['cf'][k]['mean'], 4) for k in R['cf']})
    print('labels', {v: {k: round(R['labels'][v][k]['diff'], 4) for k in ['dice', 'ave']} for v in R['labels']})
