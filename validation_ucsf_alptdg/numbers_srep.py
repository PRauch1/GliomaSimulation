# -*- coding: utf-8 -*-
"""Collect all numbers used in the Scientific Reports text, tables and legends into data/ucsf/numbers_srep.json.
Primary run: UCSF_RUN (default infer_flair; FLAIR input, N4); sensitivity run: data/ucsf/tumor_results_t2.json."""
import json, os
import numpy as np, pandas as pd
B = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get('UCSF_RUN', 'infer_flair')
R = json.load(open(os.path.join(B, 'data/ucsf/tumor_results.json')))
R2 = json.load(open(os.path.join(B, 'data/ucsf/tumor_results_t2.json')))
CF = json.load(open(os.path.join(B, 'data/ucsf/cf_results.json')))
SG = json.load(open(os.path.join(B, 'data/ucsf/tumor_subgroups.json')))
NUM = json.load(open(os.path.join(B, 'data/new/numbers.json')))
N = {}
from decimal import Decimal, ROUND_HALF_UP
def _fmt(x, k):
    t = str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-k), rounding=ROUND_HALF_UP)) if k > 0 else \
        str(Decimal(repr(float(x))).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    if t.startswith('-') and float(t) == 0: t = t[1:]
    return t.replace('-', '−')
f3 = lambda x: _fmt(x, 3); f2 = lambda x: _fmt(x, 2); f1 = lambda x: _fmt(x, 1); f0 = lambda x: _fmt(x, 0)
def ci(c, f=f3): return f'{f(c[0])}–{f(c[1])}' if not f(c[0]).startswith('−') else f'{f(c[0])} to {f(c[1])}'
def sg(x, f=f3): return (('+' if x > 0 else '') + f(x)) if round(x, 3) != 0 else f(abs(x))
def p_fmt(p):
    if p < 0.001: return '< .001'
    return ('= ' + f'{p:.3f}'.lstrip('0')) if p < 0.01 else ('= ' + f'{p:.2f}'.lstrip('0'))

d = pd.read_csv(os.path.join(B, 'data/ucsf', RUN, 'per_patient.csv')).drop_duplicates('id').set_index('id')
n = len(d); N['n_val'] = str(n)

# ---------------- Table 2 cells ----------------
def cells(prefix, r):
    N[f'{prefix}_dice'] = f"{f3(r['dice']['mean'])} ± {f3(r['dice']['sd'])} ({f3(r['dice']['median'])})"
    N[f'{prefix}_hd95'] = f"{f2(r['hd95']['mean'])} ({f2(r['hd95']['median'])})"
    N[f'{prefix}_nsd1'] = f3(r['nsd1']['mean']); N[f'{prefix}_nsd2'] = f3(r['nsd2']['mean'])
    N[f'{prefix}_ave'] = f"{f1(r['ave']['mean'])} ({f1(r['ave']['median'])})"
cells('T2_cf', R['cf'])
for k, v in [('true', 'T2_true'), ('all_perm', 'T2_perm'), ('cortex_swap', 'T2_swap')]:
    cells(v, R[k]); N[f'{v}_bias'] = f"{R[k]['bias']['mean']:+.1f} ({f1(R[k]['bias_loa'][0])} to {f1(R[k]['bias_loa'][1])})".replace('-', '−')
bcf = (d.tumor_cf_vol_pred - d.tumor_cf_vol_gt) / 1000
N['T2_cf_bias'] = f"{bcf.mean():+.1f} ({f1(bcf.mean() - 1.96 * bcf.std(ddof=1))} to {f1(bcf.mean() + 1.96 * bcf.std(ddof=1))})".replace('-', '−')

# ---------------- main comparison ----------------
t = R['true']; vc = t['vs_cf']
N.update({
    'cf_dice': f3(R['cf']['dice']['mean']), 'cf_dice_ci': ci(R['cf']['dice']['ci']), 'cf_hd95': f2(R['cf']['hd95']['mean']),
    'cf_ave': f1(R['cf']['ave']['mean']), 'cf_nsd1': f3(R['cf']['nsd1']['mean']), 'cf_nsd2': f3(R['cf']['nsd2']['mean']),
    'cf_dice_med': f3(R['cf']['dice']['median']),
    'm_dice': f3(t['dice']['mean']), 'm_dice_ci': ci(t['dice']['ci']), 'm_hd95': f2(t['hd95']['mean']), 'm_ave': f1(t['ave']['mean']),
    'm_nsd1': f3(t['nsd1']['mean']), 'm_nsd2': f3(t['nsd2']['mean']), 'm_dice_med': f3(t['dice']['median']),
    'd_dice': sg(vc['dice']['diff']), 'd_dice_ci': ci(vc['dice']['ci']), 'd_dice_abs': f3(abs(vc['dice']['diff'])),
    'd_dice_p': p_fmt(vc['dice']['p']), 'n_better_dice': str(vc['dice']['n_model_better']),
    'd_hd95': f1(vc['hd95']['diff']), 'd_hd95_ci': ci(vc['hd95']['ci'], f1), 'd_hd95_p': p_fmt(vc['hd95']['p']),
    'd_nsd1': f3(vc['nsd1']['diff']), 'd_nsd2': f3(vc['nsd2']['diff']), 'd_nsd1_p': p_fmt(vc['nsd1']['p']),
    'd_nsd2_p': p_fmt(vc['nsd2']['p']),
    'd_ave': f1(vc['ave']['diff']), 'd_ave_ci': ci(vc['ave']['ci'], f1), 'd_ave_p': p_fmt(vc['ave']['p']),
    'n_better_ave': str(vc['ave']['n_model_better']),
    'm_bias': f"{t['bias']['mean']:+.1f}".replace('-', '−'), 'm_loa': f"{f1(t['bias_loa'][0])} to {f1(t['bias_loa'][1])}",
    'cf_bias': f"{bcf.mean():+.1f}".replace('-', '−'),
    'm_r': f3(t['r']), 'm_dice_raw': f3(t['dice_raw']['mean']),
    'm_pchg': f1(t['pred_change']['median']), 'm_pchg_iqr': f"{f1(t['pred_change']['q1'])}–{f1(t['pred_change']['q3'])}"
    if t['pred_change']['q1'] >= 0 else f"{f1(t['pred_change']['q1'])} to {f1(t['pred_change']['q3'])}",
    'obs_chg': f1(R['obs_change_cm3']['median']), 'obs_chg_iqr': f"{f1(R['obs_change_cm3']['q1'])} to {f1(R['obs_change_cm3']['q3'])}",
    'n_decrease': str(R['n_decrease']), 'n_increase': str(n - R['n_decrease']),
    'pct_decrease': f"{100 * R['n_decrease'] / n:.1f}",
    'rho_abs': f2(t['spearman_abs'][0]), 'rho_abs_p': p_fmt(t['spearman_abs'][1]), 'rho_rel': f2(t['spearman_rel'][0]),
    'rho_rel_p': p_fmt(t['spearman_rel'][1]),
    'auc_abs': f2(t['auc_abs']), 'auc_abs_ci': ci(t['auc_abs_ci'], f2), 'auc_rel': f2(t['auc_rel']), 'auc_rel_ci': ci(t['auc_rel_ci'], f2),
    'new_dice': f2(t['new_dice']['mean']), 'new_prec': f"{100 * t['new_precision_pooled']:.0f}", 'new_rec': f"{100 * t['new_recall_pooled']:.0f}",
    'new_pred': f1(t['new_pred_cm3']['median']),
    'ece_raw': f3(R['calibration_raw']['ece']), 'ece_enf': f3(R['calibration_enf']['ece']),
})
# share of patients in whom the model predicted any growth
pchg = (d.tumor_true_vol_pred - d.tumor_cf_vol_pred) / 1000
N['n_pred_growth'] = str(int((pchg > 0).sum()))
# label controls
L = R['labels']
N.update({'perm_d': sg(L['all_perm']['dice']['diff']), 'perm_ci': ci(L['all_perm']['dice']['ci']),
          'perm_p': p_fmt(L['all_perm']['dice']['p']), 'perm_dice': f3(R['all_perm']['dice']['mean']),
          'swap_d': f3(L['cortex_swap']['dice']['diff']), 'swap_ci': ci(L['cortex_swap']['dice']['ci']),
          'swap_p': p_fmt(L['cortex_swap']['dice']['p']), 'swap_dice': f3(R['cortex_swap']['dice']['mean'])})
fv = pd.read_excel(os.path.join(B, 'data/new/feat_val.xlsx')); fv['id'] = fv.id.str[8:].astype(int)
dd_ = d.reset_index().merge(fv[['id', 'cortex']], on='id'); iso = dd_.cortex == 'Isocortex'
dv = (dd_.tumor_cortex_swap_vol_pred - dd_.tumor_true_vol_pred) / 1000; ddc = dd_.tumor_true_dice - dd_.tumor_cortex_swap_dice
N.update({'swap_iso_n': str(int(iso.sum())), 'swap_iso_dv': f1(dv[iso].mean()), 'swap_iso_dv_med': f1(dv[iso].median()),
          'swap_iso_better': str(int((ddc[iso] > 0).sum())), 'swap_iso_loss': f3(ddc[iso].mean()),
          'swap_oth_n': str(int((~iso).sum())), 'swap_oth_dv_abs': f1(-dv[~iso].mean()), 'swap_oth_dv_med': f1(dv[~iso].median()),
          'swap_oth_better': str(int((ddc[~iso] > 0).sum())), 'swap_oth_worse': str(int((ddc[~iso] < 0).sum())),
          'swap_oth_gain': f3(-ddc[~iso].mean()),
          'perm_changed': str(int((d.tumor_all_perm_changed_voxels > 0).sum())),
          'perm_changed_med': f1((d.tumor_all_perm_changed_voxels[d.tumor_all_perm_changed_voxels > 0] / 1000).median()),
          'swap_changed_med': f1((d.tumor_cortex_swap_changed_voxels / 1000).median())})
from scipy import stats as _st
_bl = d.tumor_cf_vol_pred / 1000; _pch = (d.tumor_true_vol_pred - d.tumor_cf_vol_pred) / 1000; _obs = (d.tumor_cf_vol_gt - d.tumor_cf_vol_pred) / 1000
_r1 = _st.spearmanr(_pch, _bl); _r2 = _st.spearmanr(_obs, _bl)
_terc = pd.qcut(_bl, 3, labels=False)
N.update({'rho_pch_bl': f2(_r1.statistic), 'rho_pch_bl_p': p_fmt(_r1.pvalue), 'rho_obs_bl': f2(_r2.statistic), 'rho_obs_bl_p': p_fmt(_r2.pvalue),
          'grow_t1': f"{100 * (_obs[_terc == 0] > 0).mean():.0f}", 'grow_t3': f"{100 * (_obs[_terc == 2] > 0).mean():.0f}",
          'grow_t2': f"{100 * (_obs[_terc == 1] > 0).mean():.0f}"})
N['_swap'] = {'iso_dv': float(dv[iso].mean()), 'iso_dd': float(-ddc[iso].mean()), 'oth_dv': float(dv[~iso].mean()),
              'oth_dd': float(-ddc[~iso].mean()), 'iso_better': int((ddc[iso] > 0).sum()), 'oth_better': int((ddc[~iso] > 0).sum())}
# lesion-mask no-change reference (dataset reference values, Supplementary Table S2)
N.update({'les_cf_dice': f3(CF['cf']['dice']['mean']), 'les_cf_hd95': f2(CF['cf']['hd95_mm']['mean'])})
lc = CF['cf']; tc = CF['tumor_cf']
def row4(r, ave, bias):
    return [f"{f3(r['dice']['mean'])} ± {f3(r['dice']['sd'])} ({f3(r['dice']['median'])})", f"{f2(r['hd95_mm']['mean'])} ({f2(r['hd95_mm']['median'])})",
            f3(r['nsd1']['mean']), f3(r['nsd2']['mean']), ave, bias]
vcf = NUM['validation']['cf']
N['S2_lesion'] = row4(lc, f"{f1(vcf['mae'])} ({f1(vcf['median_ae'])})", f"{vcf['bias']:+.1f} ({f1(vcf['loa'][0])} to {f1(vcf['loa'][1])})".replace('-', '−'))
N['S2_tumor'] = row4(tc, N['T2_cf_ave'], N['T2_cf_bias'])

# ---------------- cavity ----------------
C = R.get('cavity')
if C:
    N.update({'cav_fp_med': f1(C['fp_cm3']['median']), 'cav_fp_iqr': f"{f1(C['fp_cm3']['q1'])}–{f1(C['fp_cm3']['q3'])}",
              'cav_fpin_med': f1(C['fp_in_cav_cm3']['median']), 'cav_share_pooled': f"{100 * C['pooled_share']:.1f}",
              'cav_share_med': f"{100 * C['share_per_patient']['median']:.1f}", 'cav_n_half': str(C['n_share_gt_half'])})
    if 'nocav' in C:
        nc = C['nocav']
        N.update({'nocav_dice': f3(nc['dice']['mean']), 'nocav_sd': f3(nc['dice']['sd']), 'nocav_d': sg(nc['vs_cf']['diff']), 'nocav_ci': ci(nc['vs_cf']['ci']),
                  'nocav_p': p_fmt(nc['vs_cf']['p']), 'nocav_better': str(nc['vs_cf']['n_model_better']),
                  'nocav_gain': f3(nc['vs_true']['diff'])})
# ---------------- threshold sweep ----------------
TS = R.get('threshold_sweep')
if TS:
    best = max(TS, key=lambda k: TS[k]['dice']['mean'])
    N.update({'thr_best': best, 'thr_best_dice': f3(TS[best]['dice']['mean']), 'thr_best_d': sg(TS[best]['vs_cf']['diff']),
              'thr_best_ci': ci(TS[best]['vs_cf']['ci']), 'thr_best_p': p_fmt(TS[best]['vs_cf']['p']),
              'thr_any_above': str(any(TS[k]['vs_cf']['diff'] > 0 for k in TS)),
              'thr_max': list(TS)[-1], 'thr_max_dice': f3(TS[list(TS)[-1]]['dice']['mean'])})
    for k, v in TS.items():
        kk = 'thr' + k.replace('.', 'p')
        N.update({f'{kk}_dice': f3(v['dice']['mean']), f'{kk}_d': sg(v['vs_cf']['diff']), f'{kk}_ci': ci(v['vs_cf']['ci']),
                  f'{kk}_p': p_fmt(v['vs_cf']['p']), f'{kk}_better': str(v['vs_cf']['n_model_better']),
                  f'{kk}_newmed': f2(v['new_pred_cm3_median']),
                  f'{kk}_prec': f"{100 * v['new_precision_pooled']:.0f}" if v['new_precision_pooled'] is not None else '–',
                  f'{kk}_rec': f"{100 * v['new_recall_pooled']:.1f}"})
    _d99 = d['tumor_true_thr0.99_dice'] - d['tumor_cf_dice']
    N.update({'thr099_hi': str(int((_d99 > 0).sum())), 'thr099_lo': str(int((_d99 < 0).sum())), 'thr099_eq': str(int((_d99 == 0).sum()))})
    N['_thr_any_above'] = any(TS[k]['vs_cf']['diff'] > 0 for k in TS)
    N['_thr_ns_from'] = next((k for k in TS if TS[k]['vs_cf']['p'] >= 0.05), None)
    N['THR_NS'] = N['_thr_ns_from'] or 'none'
    N['S3_thr_rows'] = [[k, f"{f3(v['dice']['mean'])} ± {f3(v['dice']['sd'])}", ci(v['dice']['ci']), sg(v['vs_cf']['diff']),
                         ci(v['vs_cf']['ci']), str(v['vs_cf']['n_model_better']), f2(v['new_pred_cm3_median']),
                         (f"{100 * v['new_precision_pooled']:.0f}" if v['new_precision_pooled'] is not None else '–'),
                         f"{100 * v['new_recall_pooled']:.1f}"] for k, v in TS.items()]
# ---------------- agreement with the earlier evaluation ----------------
A = R.get('agreement')
if A:
    N.update({'agr_rerun': f3(A['rerun_dice']['mean']), 'agr_earlier': f3(A['earlier_dice']['mean']),
              'agr_diff': sg(A['diff']['mean']), 'agr_absdiff_med': f3(A['abs_diff']['median']),
              'agr_absdiff_mean': f3(A['abs_diff']['mean']),
              'agr_absdiff_iqr': f"{f3(A['abs_diff']['q1'])}–{f3(A['abs_diff']['q3'])}", 'agr_n005': str(A['n_absdiff_gt_005']),
              'agr_r': f2(A['pearson_r']), 'agr_volratio_med': f2(A['vol_ratio']['median']),
              'agr_growth_rerun': f1(A['growth_rerun_cm3']['median']), 'agr_growth_earlier': f1(A['growth_earlier_cm3']['median']),
              'agr_lescf': f3(A['lesion_cf_dice']['mean']), 'agr_earlier_d': sg(A['earlier_vs_cf']['diff']),
              'agr_earlier_ci': ci(A['earlier_vs_cf']['ci']), 'agr_earlier_better': str(A['earlier_vs_cf']['n_model_better']),
              'agr_rerun_d': sg(A['rerun_vs_cf']['diff']),
              'agr_earlier_dabs': f3(-A['earlier_vs_cf']['diff']),
              'agr_earlier_ci_pos': ci([-A['earlier_vs_cf']['ci'][1], -A['earlier_vs_cf']['ci'][0]]),
              'agr_gt_identical': str(A['gt_identical_n']), 'agr_gt_maxdiff': f0(A['gt_max_abs_diff_voxels'])})
# ---------------- T2 sensitivity run ----------------
t2 = R2['true']; v2 = t2['vs_cf']
N.update({'t2_dice': f3(t2['dice']['mean']), 't2_sd': f3(t2['dice']['sd']), 't2_d': sg(v2['dice']['diff']), 't2_ci': ci(v2['dice']['ci']),
          't2_better': str(v2['dice']['n_model_better']), 't2_ave': f1(t2['ave']['mean']), 't2_pchg': f1(t2['pred_change']['median']),
          't2_auc': f2(t2['auc_abs']), 't2_perm_d': sg(R2['labels']['all_perm']['dice']['diff']),
          't2_newv': f2(t2['new_pred_cm3']['median']), 't2_prec': f"{100 * t2['new_precision_pooled']:.0f}",
          't2_rec': f"{100 * t2['new_recall_pooled']:.1f}", 't2_cf': f3(R2['cf']['dice']['mean'])})
# ---------------- Fig. 3 patients (rule-based selection, fig3_select.py) ----------------
_va = pd.read_csv(os.path.join(B, 'data/new/val_a.csv'), sep=';')
_civ = pd.to_numeric(_va.loc[_va.timepoint == 2].set_index('id')['Days from 1st scan to 2nd scan'])
for _k, _i in zip(['r1', 'r2', 'r3'], [100158, 100147, 100036]):
    _r = d.loc[_i]
    N.update({f'f3{_k}_id': str(_i), f'f3{_k}_day': str(int(_civ.loc[_i])), f'f3{_k}_bl': f1(_r.tumor_cf_vol_pred / 1000),
              f'f3{_k}_fu': f1(_r.tumor_cf_vol_gt / 1000), f'f3{_k}_pred': f1(_r.tumor_true_vol_pred / 1000),
              f'f3{_k}_cf': f3(_r.tumor_cf_dice), f'f3{_k}_m': f3(_r.tumor_true_dice)})
# ---------------- T2 input with cavity analysis (full re-run, true labels) ----------------
_p = os.path.join(B, 'data/ucsf/tumor_results_t2cav.json')
if os.path.exists(_p):
    R3 = json.load(open(_p)); C3 = R3['cavity']
    N.update({'t2_cav_share': f"{100 * C3['pooled_share']:.0f}", 't2_cav_fp_med': f1(C3['fp_cm3']['median']),
              't2_cav_fpin_med': f1(C3['fp_in_cav_cm3']['median']),
              't2_nocav_dice': f3(C3['nocav']['dice']['mean']), 't2_nocav_sd': f3(C3['nocav']['dice']['sd']), 't2_nocav_d': sg(C3['nocav']['vs_cf']['diff']),
              't2_nocav_ci': ci(C3['nocav']['vs_cf']['ci']), 't2_nocav_better': str(C3['nocav']['vs_cf']['n_model_better']),
              't2_check_dice': f3(R3['true']['dice']['mean'])})
# ---------------- subgroups (Supplementary Table S1) ----------------
N['_sub'] = SG
_wt = SG['idh=IDH-wildtype']; N.update({'sg_wt_m': f3(_wt['dice_m']), 'sg_wt_cf': f3(_wt['dice_cf']), 'sg_wt_better': str(_wt['n_better']),
                                       'sg_wt_ae_m': f1(_wt['ae_m']), 'sg_wt_ae_cf': f1(_wt['ae_cf'])})
_inc = SG['chg=Increase']; N.update({'sg_inc_better': str(_inc['n_better']), 'sg_inc_n': str(_inc['n']),
                                     'sg_dec_better': str(SG['chg=Decrease or no change']['n_better'])})
N['vq'] = [f1(x) for x in SG['_vol_quartile_cuts_cm3']]

# ---------------- clinical secondary analyses (clinical_srep.py) ----------------
def iqr_(q, f=f1):
    return f"{f(q[0])}–{f(q[1])}" if not f(q[0]).startswith('−') else f"{f(q[0])} to {f(q[1])}"
def auc_(r): return f2(r['auc']), ci(r['ci'], f2)
CL = json.load(open(os.path.join(B, 'data/ucsf/clinical_endpoints.json')))
tx, G, AN, PR, IV = CL['treatment'], CL['groups'], CL['annotation'], CL['progression'], CL['interval']
N.update({'tx_rt': str(tx['rt_before']), 'tx_rt_pct': f"{100 * tx['rt_before'] / n:.1f}", 'tx_chemo': str(tx['chemo_before']),
          'tx_chemo_pct': f"{100 * tx['chemo_before'] / n:.1f}", 'tx_neither': str(tx['neither_before']),
          'tx_nort': str(n - tx['rt_before']), 'chk_rt_consistent': str(tx['rt_before_consistent']),
          'rt_conc': str(tx['rt_conc']), 'rt_conc_n': str(tx['rt_conc_n']), 'rt_placeholder': str(tx['rt_placeholder'])})
TZ = CL['temozolomide']
N.update({'tmz_n': str(TZ['before']), 'tmz_pct': f"{100 * TZ['before'] / n:.1f}", 'tmz_rt': str(TZ['before_with_rt']),
          'tmz_years': f1(TZ['years_since_start_median']), 'rt_years': f1(TZ['rt_years_since_start_median']),
          'an_dec_n': str(TZ['annotated_decrease']), 'an_dec_tmzrt': str(TZ['annotated_decrease_after_tmz_rt']),
          'dec_tmz_pct': f"{TZ['vol_decrease_tmz_pct']:.0f}", 'dec_notmz_pct': f"{TZ['vol_decrease_no_tmz_pct']:.0f}",
          'n_notmz': str(TZ['n_no_tmz'])})
EX = CL['exclusion']; assert EX['included_equals_cohort'] and EX['included'] == n
N.update({'ex_total': str(EX['n_total']), 'ex_ontx': str(EX['on_treatment']), 'ex_unk': str(EX['treatment_unknown']),
          'ex_g4': str(EX['grade4_remaining']), 'ex_all': str(EX['on_treatment'] + EX['treatment_unknown'] + EX['grade4_remaining']),
          'ex_g4_nograde': str(EX['grade4_missing_grade'])})
for key, g in [('rt', 'rt_before'), ('nort', 'no_rt_before'), ('neither', 'neither_before'), ('aninc', 'inc'),
               ('annoinc', 'no_inc'), ('annc', 'nochange')]:
    x = G[g]
    N.update({f'g_{key}_n': str(x['n']), f'g_{key}_cf': f3(x['dice_cf']), f'g_{key}_m': f3(x['dice_m']), f'g_{key}_d': sg(x['diff']),
              f'g_{key}_ci': ci(x['diff_ci']), f'g_{key}_better': str(x['n_better']), f'g_{key}_hd95': f2(x['hd95_cf']),
              f'g_{key}_hd95_med': f1(x['hd95_cf_median'])})
N.update({'an_inc_n': str(AN['n_inc']), 'an_inc_pct': f"{100 * AN['n_inc'] / n:.1f}", 'an_inc_snfh': str(AN['n_inc_snfh']),
          'an_inc_et': str(AN['n_inc_et']), 'an_nc_n': str(AN['n_nochange']), 'an_dec_only': str(AN['n_dec_only']),
          'an_inc_obs_med': f1(AN['obs_chg_inc'][0]), 'an_inc_obs_iqr': iqr_(AN['obs_chg_inc'][1:]),
          'an_nc_obs_med': f1(AN['obs_chg_nochange'][0]), 'an_nc_obs_iqr': iqr_(AN['obs_chg_nochange'][1:]),
          'an_nc_obs_range': iqr_(AN['obs_chg_nochange_range']), 'an_nc_gt1': str(AN['obs_abs_chg_nochange_gt1']),
          'an_nc_obs_absmax': f1(max(abs(AN['obs_chg_nochange_range'][0]), abs(AN['obs_chg_nochange_range'][1]))),
          'an_volinc_noinc': str(AN['vol_increase_without_annotated_increase'])})
N['an_auc_abs'], N['an_auc_abs_ci'] = auc_(AN['auc_pred_abs']); N['an_auc_rel'], N['an_auc_rel_ci'] = auc_(AN['auc_pred_rel'])
N['an_auc_bl'], N['an_auc_bl_ci'] = auc_(AN['auc_baseline'])
N.update({'pr_n': str(PR['n_documented']), 'pr_med': f0(PR['median_days']), 'pr_1y': str(PR['n_1y']),
          'pr_in_iv': str(PR['n_within_interval']), 'pr_neg': str(PR['n_nonpositive']), 'pr_60': str(PR['n_1y_within60_after_fu'])})
N['pr_auc_abs'], N['pr_auc_abs_ci'] = auc_(PR['auc_pred_abs']); N['pr_auc_rel'], N['pr_auc_rel_ci'] = auc_(PR['auc_pred_rel'])
N['pr_auc_bl'], N['pr_auc_bl_ci'] = auc_(PR['auc_baseline']); N['pr_auc_obs'], N['pr_auc_obs_ci'] = auc_(PR['auc_obs_abs_followup'])
N.update({'iv_pabs': f2(IV['pred_abs'][0]), 'iv_pabs_p': p_fmt(IV['pred_abs'][1]), 'iv_prel': f2(IV['pred_rel'][0]),
          'iv_prel_p': p_fmt(IV['pred_rel'][1]), 'iv_oabs': f2(IV['obs_abs'][0]), 'iv_oabs_p': p_fmt(IV['obs_abs'][1]),
          'iv_orel': f2(IV['obs_rel'][0]), 'iv_orel_p': p_fmt(IV['obs_rel'][1])})
N['an_dauc'] = sg(AN['auc_pred_minus_baseline']['diff'], f2); N['an_dauc_ci'] = ci(AN['auc_pred_minus_baseline']['ci'], f2)
N['pr_dauc'] = sg(PR['auc_pred_minus_baseline']['diff'], f2); N['pr_dauc_ci'] = ci(PR['auc_pred_minus_baseline']['ci'], f2)
CR = CL['criterion']
assert CR['n_model_prec_above'] == CR['n_model_better']
N.update({'crit_pct': f"{100 * CR['mean']:.0f}", 'crit_n': str(CR['n_model_prec_above']),
          'crit_min': f"{100 * CR['min']:.0f}", 'crit_max': f"{100 * CR['max']:.0f}",
          'm_prec_med': f"{100 * CR['model_prec_median']:.0f}"})
# mean diameter +4 mm per year (Mandonnet 2003) -> margin (radius) about 2 mm per year; advance over the median interval
# mean diameter growth of untreated grade 2 gliomas: 4.1 mm/year (Mandonnet 2003) and 5.8 mm/year (Pallud 2013);
# the margin (radius) advances by half of it over the median interval
_ivm = float(SG['_interval_median_days']) / 365.25
N['adv_mm'] = f2(4.1 / 2 * _ivm); N['adv_mm_hi'] = f2(5.8 / 2 * _ivm)
_blm = float((d.tumor_cf_vol_pred / 1000).median()); N['bl_med'] = f1(_blm)
_r = (3 * _blm * 1000 / (4 * np.pi)) ** (1 / 3)   # radius in mm of a sphere of the median baseline volume
_shell = lambda dr: 4 / 3 * np.pi * ((_r + dr) ** 3 - _r ** 3) / 1000
N['shv_lo'] = f1(_shell(4.1 / 2 * _ivm)); N['shv_hi'] = f1(_shell(5.8 / 2 * _ivm))

# ---------------- spatial ranking and time probe (spatial_srep.py) ----------------
_sp = os.path.join(B, 'data/ucsf/spatial_results.json')
if os.path.exists(_sp):
    SP = json.load(open(_sp)); A_ = SP['auc']
    N.update({'sp_n': str(SP['n']), 'sp_shell': str(SP['shell_mm']), 'sp_n_auc': str(A_['model_true']['n']),
              'sp_auc_m': f2(A_['model_true']['mean']), 'sp_auc_m_ci': ci(A_['model_true']['ci'], f2),
              'sp_auc_prox': f2(A_['proximity']['mean']), 'sp_auc_prox_ci': ci(A_['proximity']['ci'], f2),
              'sp_auc_int': f2(A_['intensity']['mean']), 'sp_auc_int_ci': ci(A_['intensity']['ci'], f2),
              'sp_auc_swap': f2(A_['model_swap']['mean']), 'sp_auc_perm': f2(A_['model_perm']['mean']),
              'sp_dprox': f3(SP['auc_prox_minus_model']['mean']), 'sp_dprox_ci': ci(SP['auc_prox_minus_model']['ci']),
              'sp_dprox_p': p_fmt(SP['auc_prox_minus_model']['p']), 'sp_prox_better': str(SP['auc_prox_minus_model']['n_pos']),
              'sp_dswap': sg(SP['auc_swap_minus_true']['mean']), 'sp_dswap_ci': ci(SP['auc_swap_minus_true']['ci']),
              'sp_dperm': sg(SP['auc_perm_minus_true']['mean']), 'sp_dperm_ci': ci(SP['auc_perm_minus_true']['ci']),
              'sp_or_m': f2(SP['oracle_newdice']['model_true']['mean']), 'sp_or_prox': f2(SP['oracle_newdice']['proximity']['mean']),
              'sp_or_int': f2(SP['oracle_newdice']['intensity']['mean']),
              'sp_or_d': sg(SP['oracle_prox_minus_model']['mean'], f3), 'sp_or_ci': ci(SP['oracle_prox_minus_model']['ci']),
              'sp_or_p': p_fmt(SP['oracle_prox_minus_model']['p']),
              'sp_ord_m': f3(SP['oracle_dice']['model_true']['mean']), 'sp_ord_prox': f3(SP['oracle_dice']['proximity']['mean']),
              'sp_ord_prox_d': sg(SP['oracle_dice_prox_minus_cf']['mean']), 'sp_ord_prox_ci': ci(SP['oracle_dice_prox_minus_cf']['ci']),
              'sp_ord_m_d': sg(SP['oracle_dice_model_minus_cf']['mean']), 'sp_ord_m_ci': ci(SP['oracle_dice_model_minus_cf']['ci']),
              'sp_or_m_pct': f"{100 * SP['oracle_newdice']['model_true']['mean']:.0f}",
              'sp_or_m_crit': str(SP['oracle_meets_criterion']['model_true']), 'sp_or_prox_crit': str(SP['oracle_meets_criterion']['proximity']),
              'sp_or_prox_pct': f"{100 * SP['oracle_newdice']['proximity']['mean']:.0f}",
              'sp_mt_m': f2(SP['matched']['newdice_model']['mean']), 'sp_mt_prox': f2(SP['matched']['newdice_prox']['mean']),
              'sp_mt_d': sg(SP['matched']['diff_newdice']['mean']), 'sp_mt_ci': ci(SP['matched']['diff_newdice']['ci']),
              'sp_mt_p': p_fmt(SP['matched']['diff_newdice']['p']),
              'sp_t0_med': f1(SP['t0']['t0_med']), 'sp_t0_iqr': iqr_(SP['t0']['t0_q']), 'sp_t_med': f1(SP['t0']['t_med']),
              'sp_t0_npos': str(SP['t0']['n_t0_pos']),
              'sp_t0_npos_txt': (f"all {SP['n']}" if SP['t0']['n_t0_pos'] == SP['n'] else f"{SP['t0']['n_t0_pos']} of {SP['n']}"), 'sp_t0_share': f"{100 * SP['t0']['share_pooled']:.0f}",
              'sp_t0_share_med': f"{100 * SP['t0']['share_median']:.0f}", 'sp_t0_share_iqr': f"{100 * SP['t0']['share_q'][0]:.0f}–{100 * SP['t0']['share_q'][1]:.0f}",
              'sp_t0_soft': f"{100 * SP['t0']['soft_share_pooled']:.0f}",
              'sp_new_in_shell': f"{100 * SP['new_in_shell_share']:.1f}"})
    N['S4_rows'] = [[lab, f"{f3(A_[k]['mean'])} ({ci(A_[k]['ci'])})", f3(SP['oracle_newdice'][k]['mean']), f3(SP['oracle_dice'][k]['mean'])]
                    for k, lab in [('model_true', 'Model, correct labels'), ('model_swap', 'Model, cortical architecture swapped'),
                                   ('model_perm', 'Model, labels permuted'), ('proximity', 'Distance from the baseline tumor'),
                                   ('intensity', 'FLAIR intensity')]]
# ---------------- analyses added after independent review ----------------
# spatial ranking by annotated change; interval and annotated increase; tumors without histomorphologic specification
from sklearn.metrics import roc_auc_score as _auc
_cl = pd.read_csv(os.path.join(B, 'data/ucsf/clinical_per_patient.csv')).set_index('id').sort_index()
_inc = _cl.annotated_increase.astype(bool); _nc = _cl.annotated_no_change.astype(bool)
if os.path.exists(os.path.join(B, 'data/ucsf/spatial_flair/per_patient.csv')):
    _s = pd.read_csv(os.path.join(B, 'data/ucsf/spatial_flair/per_patient.csv')).set_index('id').sort_index()
    assert (_s.index == _cl.index).all()
    _dpm = _s.auc_proximity - _s.auc_model_true
    N.update({'sp_prox_inc': f2(_s.auc_proximity[_inc].mean()), 'sp_prox_nc': f2(_s.auc_proximity[_nc].mean()),
              'sp_m_inc': f2(_s.auc_model_true[_inc].mean()), 'sp_m_nc': f2(_s.auc_model_true[_nc].mean()),
              'sp_prox_grp_p': p_fmt(_st.mannwhitneyu(_s.auc_proximity[_inc], _s.auc_proximity[_nc]).pvalue),
              'sp_dprox_inc': sg(_dpm[_inc].mean()), 'sp_dprox_inc_n': str(int((_dpm[_inc] > 0).sum())),
              'sp_dprox_inc_p': p_fmt(_st.wilcoxon(_dpm[_inc]).pvalue),
              'sp_nc_newmed': f1((_s.new_total[_nc] / 1000).median())})
_va = pd.read_csv(os.path.join(B, 'data/new/val_a.csv'), sep=';')
_iv = pd.to_numeric(_va.loc[_va.timepoint == 2].set_index('id')['Days from 1st scan to 2nd scan']).loc[_cl.index]
N.update({'iv_inc_med': f0(_iv[_inc].median()), 'iv_noinc_med': f0(_iv[~_inc].median()),
          'iv_inc_p': p_fmt(_st.mannwhitneyu(_iv[_inc], _iv[~_inc]).pvalue), 'iv_inc_auc': f2(_auc(_inc, -_iv))})
_fv = pd.read_excel(os.path.join(B, 'data/new/feat_val.xlsx')); _fv['pid'] = _fv.id.str[8:].astype(int)
_fv = _fv.set_index('pid').loc[_cl.index]
_unk = _fv.histo.isin(['Glioma NOS', 'Other', 'IDH mut NOS'])
_dd = d.sort_index(); assert (_dd.index == _cl.index).all()
_pc = (_dd.tumor_true_vol_pred - _dd.tumor_cf_vol_pred) / 1000; _oc = (_dd.tumor_cf_vol_gt - _dd.tumor_cf_vol_pred) / 1000
_df = _dd.tumor_true_dice - _dd.tumor_cf_dice
N.update({'unk_n': str(int(_unk.sum())), 'unk_pchg': f1(_pc[_unk].median()), 'kn_pchg': f1(_pc[~_unk].median()),
          'unk_ochg': f1(_oc[_unk].median()), 'kn_ochg': f1(_oc[~_unk].median()),
          'unk_d': sg(_df[_unk].mean()), 'kn_d': sg(_df[~_unk].mean()),
          'unk_pchg_p': p_fmt(_st.mannwhitneyu(_pc[_unk], _pc[~_unk]).pvalue)})
# treatment groups: frequency of volume decrease and observed change; deficit with and without prior temozolomide
for key, g in [('rt', 'rt_before'), ('nort', 'no_rt_before')]:
    x = G[g]; N[f'g_{key}_decpct'] = f"{100 * x['n_decrease'] / x['n']:.0f}"; N[f'g_{key}_obs'] = sg(x['obs_chg_med'], f1)
_tmz = _cl.temozolomide_before_baseline.astype(bool)
N['tmz_d'] = sg(_df[_tmz].mean()); N['notmz_d'] = sg(_df[~_tmz].mean())
# earlier evaluation of separately trained model variants (masks including resection cavities; data/nc/summary.csv)
_ns = os.path.join(B, 'data/nc/summary.csv')
if os.path.exists(_ns):
    _lines = open(_ns).read().splitlines(); _i = [k for k, l in enumerate(_lines) if l.startswith('Enforced')][0]
    from io import StringIO
    _e = pd.read_csv(StringIO('\n'.join(_lines[_i + 1:]))).dropna(subset=['dataset'])
    _lab = {'no film': 'Imaging only (no conditioning)', 'cortex': 'Cortical architecture', 'idh': 'IDH status',
            'histo': 'Histomorphologic type', 'idh, cortex': 'IDH status and cortical architecture',
            'histo, cortex': 'Histomorphologic type and cortical architecture', 'idh, histo': 'IDH status and histomorphologic type',
            'Idh, histo & cortex': 'All three variables (model of this study)'}
    _order = ['no film', 'cortex', 'idh', 'histo', 'idh, cortex', 'histo, cortex', 'idh, histo', 'Idh, histo & cortex']
    _e = _e.set_index('dataset').loc[_order]
    N['S5_rows'] = [[_lab[k], f"{f3(r.dice_mean_enforced)} ({f3(r.dice_median_enforced)})", f2(r.hd95_mean_mm_enforced),
                     f1(r.abs_vol_err_mean_mm3_enforced / 1000)] for k, r in _e.iterrows()]
    _sub = _e.drop(index='Idh, histo & cortex')
    N['ev_min'] = f3(_sub.dice_mean_enforced.min()); N['ev_max'] = f3(_sub.dice_mean_enforced.max())
    N['ev_full'] = f3(_e.loc['Idh, histo & cortex', 'dice_mean_enforced']); N['ev_img'] = f3(_e.loc['no film', 'dice_mean_enforced'])
    N['ev_full_ae'] = f1(_e.loc['Idh, histo & cortex', 'abs_vol_err_mean_mm3_enforced'] / 1000)
    N['ev_img_ae'] = f1(_e.loc['no film', 'abs_vol_err_mean_mm3_enforced'] / 1000)
    _lae = ((d.lesion_cf_vol_pred - d.lesion_cf_vol_gt).abs() / 1000).mean()
    _lae_m = ((d.lesion_true_vol_pred - d.lesion_true_vol_gt).abs() / 1000).mean()
    N['S5_ref'] = ['No-change reference', f"{f3(d.lesion_cf_dice.mean())} ({f3(d.lesion_cf_dice.median())})",
                   f2(d.lesion_cf_hd95.mean()), f1(_lae)]
    N['S5_rerun'] = ['All three variables, present re-run', f"{f3(d.lesion_true_dice.mean())} ({f3(d.lesion_true_dice.median())})",
                     f2(d.lesion_true_hd95.mean()), f1(_lae_m)]
json.dump(N, open(os.path.join(B, 'data/ucsf/numbers_srep.json'), 'w'), ensure_ascii=False, indent=1)
if __name__ == '__main__':
    print(json.dumps({k: v for k, v in N.items() if not k.startswith('_') and not k.startswith('S')}, ensure_ascii=False, indent=0)[:6000])
