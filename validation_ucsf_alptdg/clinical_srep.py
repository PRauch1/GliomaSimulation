# -*- coding: utf-8 -*-
"""Clinically oriented secondary analyses from existing per-patient results and the dataset's clinical table
(no additional inference). Output: data/ucsf/clinical_endpoints.json.

- treatment before baseline (radiotherapy, chemotherapy) from the clinical table
- annotated change between the two examinations (dataset change labels for the surrounding nonenhancing FLAIR
  hyperintensity and the enhancing tissue): agreement by group, discrimination of annotated increase
- documented progression within one year after baseline: discrimination
- association of predicted and observed volume change with the interval
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

B = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get('UCSF_RUN', 'infer_flair')
XL = os.path.join(B, 'data/ucsf/raw/UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx')
d = pd.read_csv(os.path.join(B, 'data/ucsf', RUN, 'per_patient.csv')).drop_duplicates('id').set_index('id').sort_index()
ci_ = pd.read_excel(XL, sheet_name='Clinical Info').set_index('SubjectID').loc[d.index]
lp = pd.read_excel(XL, sheet_name='UCSF-LPTDG')
t2 = lp[lp.Timepoint == 2].set_index('SubjectID').loc[d.index]

bl = d.tumor_cf_vol_pred / 1000; fu = d.tumor_cf_vol_gt / 1000; pr = d.tumor_true_vol_pred / 1000
pch = pr - bl; och = fu - bl
iv = pd.to_numeric(ci_['Days from 1st scan to 2nd scan'])
rt_days = ci_['Days from 1st scan to 1st RT start (neg = RT first)']      # negative: radiotherapy before baseline
ch_days = ci_['Days from 1st chemo start to 1st scan']                    # positive: chemotherapy before baseline
prog = ci_['Days from 1st scan to next progression']
rt_before = rt_days < 0; ch_before = ch_days > 0
both_rt_ch = rt_days.notna() & ch_days.notna()
tmz = ci_['1st Chemo type'].astype(str).str.contains('TMZ|temozolomide', case=False, regex=True)
tmz_before = tmz & ch_before                                       # temozolomide as first chemotherapy, started before baseline
rt_conc = both_rt_ch & ((rt_days + ch_days).abs() <= 1)          # radiotherapy and chemotherapy started on the same day
rt_placeholder = rt_days < -36525                                  # dates more than 100 years before baseline
inc = (t2.SNFH_CHANGE_OVERALL == 'Increased') | (t2.ET_CHANGE_OVERALL == 'Increased')
nochange = (t2.SNFH_CHANGE_OVERALL == 'Unchanged') & (t2.ET_CHANGE_OVERALL == 'Unchanged')
prog1y = (prog > 0) & (prog <= 365)                                  # first progression listed after the first examination
diff = d.tumor_true_dice - d.tumor_cf_dice

rng = np.random.default_rng(42)
def boot_mean(x, n=10000):
    x = np.asarray(x, float); b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def auc_ci(y, s, n=2000):
    y = np.asarray(y, int); s = np.asarray(s, float); a = roc_auc_score(y, s); b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        b.append(roc_auc_score(y[i], s[i]))
    return {'auc': float(a), 'ci': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
def group(m):
    m = np.asarray(m, bool)
    return {'n': int(m.sum()), 'dice_cf': float(d.tumor_cf_dice[m].mean()), 'dice_m': float(d.tumor_true_dice[m].mean()),
            'hd95_cf': float(d.tumor_cf_hd95[m].mean()), 'hd95_cf_median': float(d.tumor_cf_hd95[m].median()),
            'diff': float(diff[m].mean()), 'diff_ci': boot_mean(diff[m]), 'n_better': int((diff[m] > 0).sum()),
            'obs_chg_med': float(och[m].median()), 'pred_chg_med': float(pch[m].median()), 'n_decrease': int((och[m] < 0).sum())}

R = {'n': int(len(d))}
sx_days = ci_['Days from 1st surgery/DX  to 1st scan']
R['treatment'] = {'rt_conc': int(rt_conc.sum()), 'rt_conc_n': int(both_rt_ch.sum()), 'rt_placeholder': int(rt_placeholder.sum()),
                  'rt_conc_header_sign_consistent': int((rt_conc & rt_before & ch_before).sum()),
                  'rt_before_consistent': int(((-rt_days[rt_before]) <= sx_days[rt_before] + 1).sum()),
                  'rt_before': int(rt_before.sum()), 'rt_after': int((rt_days > 0).sum()), 'rt_none': int(rt_days.isna().sum()),
                  'chemo_before': int(ch_before.sum()), 'either_before': int((rt_before | ch_before).sum()),
                  'neither_before': int((~(rt_before | ch_before)).sum()),
                  'on_treatment_at_scans': int((ci_['On tx at 1st scan (c = chemo, rt, n = none)'] != 'n').sum()
                                               + (ci_['On tx at 2nd scan (c = chemo, rt, n = none)'] != 'n').sum())}
_all = pd.read_excel(XL, sheet_name='Clinical Info').set_index('SubjectID')
_o1 = _all['On tx at 1st scan (c = chemo, rt, n = none)'].astype(str).str.strip().str.lower()
_o2 = _all['On tx at 2nd scan (c = chemo, rt, n = none)'].astype(str).str.strip().str.lower()
_ontx = _o1.isin(['c', 'rt']) | _o2.isin(['c', 'rt']); _unk = ~_ontx & ~((_o1 == 'n') & (_o2 == 'n'))
_g4 = (_all.Grade == 4) | _all['WHO 2021 Diagnosis'].isin(['Glioblastoma', 'Gliosarcoma'])
_rest = ~_ontx & ~_unk
R['exclusion'] = {'n_total': int(len(_all)), 'on_treatment': int(_ontx.sum()), 'treatment_unknown': int(_unk.sum()),
                  'grade4_remaining': int((_rest & _g4).sum()), 'included': int((_rest & ~_g4).sum()),
                  'included_equals_cohort': bool(set(_all.index[_rest & ~_g4]) == set(d.index)),
                  'grade4_missing_grade': int((_rest & _g4 & _all.Grade.isna()).sum())}
_dec = och < 0
_andec = (t2.SNFH_CHANGE_OVERALL == 'Decreased') | (t2.ET_CHANGE_OVERALL == 'Decreased')
R['temozolomide'] = {'before': int(tmz_before.sum()), 'before_with_rt': int((tmz_before & rt_before).sum()),
                     'years_since_start_median': float(ch_days[tmz_before].median() / 365.25),
                     'rt_years_since_start_median': float((-rt_days[rt_before]).median() / 365.25),
                     'annotated_decrease': int(_andec.sum()), 'annotated_decrease_after_tmz_rt': int((_andec & tmz_before & rt_before).sum()),
                     'vol_decrease_tmz_pct': float(100 * _dec[tmz_before].mean()), 'vol_decrease_no_tmz_pct': float(100 * _dec[~tmz_before].mean()),
                     'n_no_tmz': int((~tmz_before).sum())}
R['groups'] = {'rt_before': group(rt_before), 'no_rt_before': group(~rt_before), 'chemo_before': group(ch_before),
               'no_chemo_before': group(~ch_before), 'neither_before': group(~(rt_before | ch_before)),
               'inc': group(inc), 'no_inc': group(~inc), 'nochange': group(nochange)}
R['annotation'] = {'n_inc': int(inc.sum()), 'n_inc_snfh': int((t2.SNFH_CHANGE_OVERALL == 'Increased').sum()),
                   'n_inc_et': int((t2.ET_CHANGE_OVERALL == 'Increased').sum()), 'n_nochange': int(nochange.sum()),
                   'n_dec_only': int((~inc & ~nochange).sum()),
                   'obs_chg_inc': [float(x) for x in och[inc].quantile([.5, .25, .75])],
                   'obs_chg_nochange': [float(x) for x in och[nochange].quantile([.5, .25, .75])],
                   'obs_chg_nochange_range': [float(och[nochange].min()), float(och[nochange].max())],
                   'obs_abs_chg_nochange_gt1': int((och[nochange].abs() > 1).sum()),
                   'vol_increase_without_annotated_increase': int(((och > 0) & ~inc).sum()),
                   'auc_pred_abs': auc_ci(inc, pch), 'auc_pred_rel': auc_ci(inc, pch / bl), 'auc_baseline': auc_ci(inc, bl)}
R['progression'] = {'n_documented': int((prog > 0).sum()), 'n_nonpositive': int((prog <= 0).sum()),
                    'median_days': float(prog[prog > 0].median()),
                    'n_1y': int(prog1y.sum()), 'n_within_interval': int(((prog > 0) & (prog <= iv)).sum()),
                    'n_1y_within60_after_fu': int((prog1y & (prog - iv <= 60)).sum()),
                    'auc_pred_abs': auc_ci(prog1y, pch), 'auc_pred_rel': auc_ci(prog1y, pch / bl), 'auc_baseline': auc_ci(prog1y, bl),
                    'auc_obs_abs_followup': auc_ci(prog1y, och)}
def auc_diff(y, s1, s2, n=2000):
    """Paired bootstrap of AUC(s1) - AUC(s2)."""
    y = np.asarray(y, int); s1 = np.asarray(s1, float); s2 = np.asarray(s2, float); b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        b.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return {'diff': float(roc_auc_score(y, s1) - roc_auc_score(y, s2)), 'ci': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
R['annotation']['auc_pred_minus_baseline'] = auc_diff(inc, pch, bl)
R['progression']['auc_pred_minus_baseline'] = auc_diff(prog1y, pch, bl)
# Adding voxels A to the baseline mask M raises Dice(M, G) iff |A ∩ G| / |A| > Dice(M, G) / 2 (Supplementary Methods)
crit = d.tumor_cf_dice / 2
prec = d['tumor_true_thr0.5_new_tp'] / d['tumor_true_thr0.5_new_pred']
R['criterion'] = {'mean': float(crit.mean()), 'min': float(crit.min()), 'max': float(crit.max()),
                  'n_model_prec_above': int((prec > crit).sum()), 'n_model_better': int((diff > 0).sum()),
                  'model_prec_median': float(prec.median())}
def sp(x, y):
    r = stats.spearmanr(x, y); return [float(r.statistic), float(r.pvalue)]
R['interval'] = {'pred_abs': sp(pch, iv), 'pred_rel': sp(pch / bl, iv), 'obs_abs': sp(och, iv), 'obs_rel': sp(och / bl, iv)}
pd.DataFrame({'rt_before_baseline': rt_before, 'chemo_before_baseline': ch_before, 'temozolomide_before_baseline': tmz_before, 'annotated_increase': inc,
              'annotated_no_change': nochange, 'progression_within_1y': prog1y,
              'change_snfh': t2.SNFH_CHANGE_OVERALL, 'change_et': t2.ET_CHANGE_OVERALL}).to_csv(
    os.path.join(B, 'data/ucsf/clinical_per_patient.csv'))
json.dump(R, open(os.path.join(B, 'data/ucsf/clinical_endpoints.json'), 'w'), indent=1)
if __name__ == '__main__':
    print(json.dumps(R, indent=1))
