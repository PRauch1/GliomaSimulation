# -*- coding: utf-8 -*-
"""Summary of the spatial and time analyses (scripts/ucsf_spatial.py output) -> data/ucsf/spatial_results.json."""
import json, os
import numpy as np, pandas as pd
from scipy import stats

B = os.path.dirname(os.path.abspath(__file__))
s = pd.read_csv(os.path.join(B, 'data/ucsf/spatial_flair/per_patient.csv')).drop_duplicates('id').set_index('id').sort_index()
d = pd.read_csv(os.path.join(B, 'data/ucsf/infer_flair/per_patient.csv')).drop_duplicates('id').set_index('id').loc[s.index]
assert (s.pred_new_t == d['tumor_true_thr0.5_new_pred']).all(), 'spatial run inconsistent with primary run'
rng = np.random.default_rng(42)
def mci(x, n=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return {'mean': float(x.mean()), 'median': float(np.median(x)), 'ci': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            'n': int(len(x)), 'sd': float(x.std(ddof=1))}
def paired(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); x = np.asarray(a[m] - b[m], float)
    r = mci(x); r['p'] = float(stats.wilcoxon(x).pvalue); r['n_pos'] = int((x > 0).sum()); r['n_neg'] = int((x < 0).sum())
    return r
K = ['model_true', 'model_swap', 'model_perm', 'proximity', 'intensity']
R = {'n': int(len(s)), 'shell_mm': 15}
R['auc'] = {k: mci(s[f'auc_{k}']) for k in K}
R['auc_prox_minus_model'] = paired(s.auc_proximity, s.auc_model_true)
R['auc_swap_minus_true'] = paired(s.auc_model_swap, s.auc_model_true)
R['auc_perm_minus_true'] = paired(s.auc_model_perm, s.auc_model_true)
R['auc_model_minus_intensity'] = paired(s.auc_model_true, s.auc_intensity)
R['oracle_newdice'] = {k: mci(s[f'oracle_newdice_{k}']) for k in K}
R['oracle_meets_criterion'] = {k: int((s[f'oracle_newdice_{k}'] > s.dice_nochange / 2).sum()) for k in K}
R['oracle_dice'] = {k: mci(s[f'oracle_dice_{k}']) for k in K}
R['oracle_prox_minus_model'] = paired(s.oracle_newdice_proximity, s.oracle_newdice_model_true)
R['oracle_dice_prox_minus_cf'] = paired(s.oracle_dice_proximity, s.dice_nochange)
R['oracle_dice_model_minus_cf'] = paired(s.oracle_dice_model_true, s.dice_nochange)
R['matched'] = {'newdice_model': mci(s.matched_newdice_model), 'newdice_prox': mci(s.matched_newdice_proximity),
                'diff_newdice': paired(s.matched_newdice_proximity, s.matched_newdice_model),
                'dice_model': mci(s.matched_dice_model), 'dice_prox': mci(s.matched_dice_proximity),
                'diff_dice': paired(s.matched_dice_proximity, s.matched_dice_model),
                'prec_model_pooled': float(((s.matched_prec_model * s.pred_new_t).sum()) / s.pred_new_t.sum()),
                'dice_prox_minus_cf': paired(s.matched_dice_proximity, s.dice_nochange)}
t, t0 = s.pred_new_t / 1000, s.pred_new_t0 / 1000
R['t0'] = {'t_med': float(t.median()), 't0_med': float(t0.median()), 't0_q': [float(t0.quantile(.25)), float(t0.quantile(.75))],
           'n_t0_pos': int((s.pred_new_t0 > 0).sum()), 'share_pooled': float(s.pred_new_t0.sum() / s.pred_new_t.sum()),
           'share_median': float((s.pred_new_t0 / s.pred_new_t).median()),
           'share_q': [float((s.pred_new_t0 / s.pred_new_t).quantile(.25)), float((s.pred_new_t0 / s.pred_new_t).quantile(.75))],
           'soft_share_pooled': float(s.soft_new_t0.sum() / s.soft_new_t.sum()),
           'n_t0_ge_half': int((s.pred_new_t0 >= 0.5 * s.pred_new_t).sum())}
R['new_in_shell_share'] = float(s.new_in_shell.sum() / max(s.new_total.sum(), 1))
R['n_no_new'] = int((s.new_total == 0).sum())
json.dump(R, open(os.path.join(B, 'data/ucsf/spatial_results.json'), 'w'), indent=1)
if __name__ == '__main__':
    print('n', R['n'], '| AUC', {k: round(v['mean'], 3) for k, v in R['auc'].items()})
    print('prox-model', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in R['auc_prox_minus_model'].items()})
    print('swap-true', round(R['auc_swap_minus_true']['mean'], 4), R['auc_swap_minus_true']['ci'], 'perm-true', round(R['auc_perm_minus_true']['mean'], 4))
    print('oracle newdice', {k: round(v['mean'], 3) for k, v in R['oracle_newdice'].items()}, 'oracle dice', {k: round(v['mean'], 3) for k, v in R['oracle_dice'].items()})
    print('oracle prox-model', round(R['oracle_prox_minus_model']['mean'], 3), R['oracle_prox_minus_model']['p'])
    print('matched', {k: (round(v['mean'], 3) if isinstance(v, dict) else round(v, 3)) for k, v in R['matched'].items()})
    print('t0', R['t0']); print('new in shell share', R['new_in_shell_share'], 'n no new', R['n_no_new'])
