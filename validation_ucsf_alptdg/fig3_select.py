# -*- coding: utf-8 -*-
"""Select three illustrative validation patients for Fig. 3 from the primary run (rule-based, documented in the legend)."""
import os
import numpy as np, pandas as pd
B = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get('UCSF_RUN', 'infer_flair')
d = pd.read_csv(os.path.join(B, 'data/ucsf', RUN, 'per_patient.csv')).drop_duplicates('id').set_index('id')
fv = pd.read_excel(os.path.join(B, 'data/new/feat_val.xlsx')); fv['pid'] = fv.id.str[8:].astype(int); fv = fv.set_index('pid')
d = d.join(fv[['cortex', 'idh', 'histo']])
d['bl'] = d.tumor_cf_vol_pred / 1000; d['fu'] = d.tumor_cf_vol_gt / 1000; d['pred'] = d.tumor_true_vol_pred / 1000
d['obs'] = d.fu - d.bl; d['pch'] = d.pred - d.bl; d['dd'] = d.tumor_true_dice - d.tumor_cf_dice
cols = ['day', 'cortex', 'idh', 'histo', 'bl', 'fu', 'pred', 'obs', 'pch', 'tumor_cf_dice', 'tumor_true_dice', 'dd', 'tumor_true_new_dice']
med_pch, med_dd = d.pch.median(), d.dd.median()
# 1 typical: stable tumor (|obs| < 1 cm3), predicted change and Dice difference closest to the cohort medians
s1 = d[(d.obs.abs() < 1)].copy(); s1['score'] = ((s1.pch - med_pch) / d.pch.std()) ** 2 + ((s1.dd - med_dd) / d.dd.std()) ** 2
# 2 growth: observed increase > 3 cm3 and model better than the reference; highest new-growth Dice
s2 = d[(d.obs > 3) & (d.dd > 0)].sort_values('tumor_true_new_dice', ascending=False)
# 3 overestimation: observed change <= 0; largest predicted increase
s3 = d[d.obs <= 0].sort_values('pch', ascending=False)
pd.set_option('display.width', 220)
print('median predicted change', round(med_pch, 2), 'median Dice difference', round(med_dd, 3))
print('typical:\n', s1.sort_values('score')[cols].head(5).round(3))
print('growth:\n', s2[cols].head(5).round(3))
print('overestimation:\n', s3[cols].head(5).round(3))
