# -*- coding: utf-8 -*-
"""Descriptive subgroups (tumor masks): no-change reference vs conditioned model.
Input: data/ucsf/<UCSF_RUN>/per_patient.csv; output: data/ucsf/<UCSF_SUB_OUT> (default tumor_subgroups.json)."""
import json, os
import numpy as np, pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get('UCSF_RUN', 'infer_flair'); OUT = os.environ.get('UCSF_SUB_OUT', 'tumor_subgroups.json')
d = pd.read_csv(os.path.join(B, 'data/ucsf', RUN, 'per_patient.csv')).drop_duplicates('id').set_index('id')
fv = pd.read_excel(os.path.join(B, 'data/new/feat_val.xlsx')); fv['pid'] = fv.id.str[8:].astype(int); fv = fv.set_index('pid').loc[d.index]
D = pd.DataFrame(index=d.index)
D['dice_cf'] = d.tumor_cf_dice; D['dice_m'] = d.tumor_true_dice
D['ae_cf'] = (d.tumor_cf_vol_pred - d.tumor_cf_vol_gt).abs() / 1000
D['ae_m'] = (d.tumor_true_vol_pred - d.tumor_true_vol_gt).abs() / 1000
D['idh'] = fv.idh.fillna('Unknown').map({'Yes': 'IDH-mutant', 'No': 'IDH-wildtype', 'Unknown': 'IDH unknown'})
D['histo_known'] = ~fv.histo.isin(['Glioma NOS', 'Other', 'IDH mut NOS'])
vi, vc = fv.idh, fv['1p19q']
D['integrated'] = np.select([(vi == 'Yes') & (vc == 'Yes'), (vi == 'Yes') & (vc == 'No'), vi == 'No'],
                            ['Oligodendroglioma', 'Astrocytoma, IDH-mutant', 'IDH-wildtype'], 'Not classifiable')
D['grade'] = fv.grade.map({1.0: 'Grade 1', 2.0: 'Grade 2', 3.0: 'Grade 3'}).fillna('Grade not documented')
D['cortex'] = fv.cortex
D['vol_q'] = pd.qcut(d.tumor_cf_vol_pred / 1000, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
D['chg'] = np.where(d.tumor_cf_vol_gt > d.tumor_cf_vol_pred, 'Increase', 'Decrease or no change')
_va = pd.read_csv(os.path.join(B, 'data/new/val_a.csv'), sep=';')
_civ = pd.to_numeric(_va.loc[_va.timepoint == 2].set_index('id')['Days from 1st scan to 2nd scan']).loc[d.index]
D['interval'] = np.where(_civ > _civ.median(), 'Interval above median', 'Interval at or below median')
_cl = pd.read_csv(os.path.join(B, 'data/ucsf/clinical_per_patient.csv')).set_index('id').loc[d.index]   # clinical_srep.py
D['rt'] = np.where(_cl.rt_before_baseline, 'RT before baseline', 'No RT before baseline')
D['chemo'] = np.where(_cl.chemo_before_baseline, 'Chemo before baseline', 'No chemo before baseline')
D['tmz'] = np.where(_cl.temozolomide_before_baseline, 'TMZ before baseline', 'No TMZ before baseline')
D['annot'] = np.where(_cl.annotated_increase, 'Annotated increase', 'No annotated increase')
sub = {}
for col in ['idh', 'histo_known', 'integrated', 'grade', 'cortex', 'vol_q', 'chg', 'annot', 'interval', 'rt', 'chemo', 'tmz']:
    for g, s in D.groupby(col, observed=True):
        sub[f'{col}={g}'] = {'n': int(len(s)), 'dice_cf': float(s.dice_cf.mean()), 'dice_m': float(s.dice_m.mean()),
                             'n_better': int((s.dice_m > s.dice_cf).sum()), 'ae_cf': float(s.ae_cf.mean()),
                             'ae_m': float(s.ae_m.mean())}
q = (d.tumor_cf_vol_pred / 1000).quantile([.25, .5, .75]).tolist()
sub['_vol_quartile_cuts_cm3'] = q
sub['_interval_median_days'] = float(_civ.median())
json.dump(sub, open(os.path.join(B, 'data/ucsf', OUT), 'w'), indent=1)
if __name__ == '__main__':
    for k, v in sub.items():
        if k.startswith('_'): print(k, v); continue
        print(f"{k:40s} n={v['n']:3d} dice cf {v['dice_cf']:.3f} m {v['dice_m']:.3f} better {v['n_better']:2d} | "
              f"AE cf {v['ae_cf']:.1f} m {v['ae_m']:.1f}")
