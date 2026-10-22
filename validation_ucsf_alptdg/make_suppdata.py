# -*- coding: utf-8 -*-
"""Supplementary Data 1: per-patient validation results (UCSF-ALPTDG, tumor masks without resection cavity)."""
import os, json
import numpy as np, pandas as pd, nibabel as nib

B = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get('UCSF_RUN', 'infer_flair')
d = pd.read_csv(os.path.join(B, 'data/ucsf', RUN, 'per_patient.csv')).drop_duplicates('id').set_index('id').sort_index()
fv = pd.read_excel(os.path.join(B, 'data/new/feat_val.xlsx')); fv['pid'] = fv.id.str[8:].astype(int); fv = fv.set_index('pid')
cfg = json.load(open(os.path.join(B, 'infer/model_config.json')))
inv = {k: {v: kk for kk, v in cfg['vocabs'][k].items()} for k in cfg['cat_order']}
NAMES = {'cortex': {'Isocortex': 'isocortical', 'Mesocortex': 'mesocortical', 'Allocortex': 'allocortical', 'WM': 'white matter',
                    'Multifokal': 'multifocal', 'Cerebellar': 'cerebellar', 'X': 'gliomatosis', '__UNK__': 'unknown'},
         'idh': {'Yes': 'mutant', 'No': 'wildtype', '__UNK__': 'unknown'},
         'histo': {'Astrocytoma': 'astrocytic', 'Oligodendroglioma': 'oligodendroglial', '__UNK__': 'unknown'}}
R = os.path.join(B, 'data/ucsf/raw')
cav = {}
for i in d.index:
    l1 = np.rint(np.asanyarray(nib.load(f'{R}/{i}/{i}_time1_seg.nii.gz').dataobj)).astype(np.int16)
    cav[i] = float((l1 == 4).sum()) / 1000
_va = pd.read_csv(os.path.join(B, 'data/new/val_a.csv'), sep=';')
civ = pd.to_numeric(_va.loc[_va.timepoint == 2].set_index('id')['Days from 1st scan to 2nd scan'])
_cl = pd.read_csv(os.path.join(B, 'data/ucsf/clinical_per_patient.csv')).set_index('id')
_sp = pd.read_csv(os.path.join(B, 'data/ucsf/spatial_flair/per_patient.csv')).drop_duplicates('id').set_index('id')
_yn = lambda x: 'yes' if bool(x) else 'no'
rows = []
for i, r in d.iterrows():
    meta = [int(x) for x in str(r.meta_true).split(',')]
    lab = {k: NAMES[k].get(inv[k][m], inv[k][m]) for k, m in zip(cfg['cat_order'], meta)}
    o = {'Patient ID (UCSF-ALPTDG)': i, 'Interval baseline to follow-up, days': int(civ.loc[i]),
         'Interval used for the time channel, days': int(r.day),
         'Cortical architecture (model input)': lab['cortex'], 'IDH status (model input)': lab['idh'],
         'Histomorphologic type (model input)': lab['histo'],
         'Baseline tumor volume, cm3': r.tumor_cf_vol_pred / 1000, 'Follow-up tumor volume, cm3': r.tumor_cf_vol_gt / 1000,
         'Baseline resection cavity volume, cm3': cav[i],
         'No-change reference: Dice': r.tumor_cf_dice, 'No-change reference: HD95, mm': r.tumor_cf_hd95,
         'No-change reference: NSD 1 mm': r.tumor_cf_nsd1, 'No-change reference: NSD 2 mm': r.tumor_cf_nsd2,
         'No-change reference: absolute volumetric error, cm3': abs(r.tumor_cf_vol_pred - r.tumor_cf_vol_gt) / 1000,
         'Model: predicted volume, cm3': r.tumor_true_vol_pred / 1000, 'Model: Dice': r.tumor_true_dice,
         'Model: HD95, mm': r.tumor_true_hd95, 'Model: NSD 1 mm': r.tumor_true_nsd1, 'Model: NSD 2 mm': r.tumor_true_nsd2,
         'Model: absolute volumetric error, cm3': abs(r.tumor_true_vol_pred - r.tumor_true_vol_gt) / 1000,
         'Model: Dice within new-growth region': r.tumor_true_new_dice,
         'Model: false-positive volume, cm3': r.tumor_true_fp / 1000,
         'Model: false-positive volume within baseline cavity, cm3': r.get('tumor_true_fp_in_bl_cavity', np.nan) / 1000,
         'Labels permuted: Dice': r.tumor_all_perm_dice, 'Labels permuted: predicted volume, cm3': r.tumor_all_perm_vol_pred / 1000,
         'Cortical architecture swapped: Dice': r.tumor_cortex_swap_dice,
         'Cortical architecture swapped: predicted volume, cm3': r.tumor_cortex_swap_vol_pred / 1000}
    c_, s_ = _cl.loc[i], (_sp.loc[i] if i in _sp.index else None)
    o.update({'Radiotherapy before baseline (dataset)': _yn(c_.rt_before_baseline),
              'Chemotherapy before baseline (dataset)': _yn(c_.chemo_before_baseline),
              'Annotated change, FLAIR hyperintensity (dataset)': str(c_.change_snfh).lower(),
              'Annotated change, enhancing tissue (dataset)': str(c_.change_et).lower(),
              'Progression documented within 365 days (dataset)': _yn(c_.progression_within_1y)})
    if s_ is not None:
        o.update({'Model: predicted new volume, actual interval, cm3': s_.pred_new_t / 1000,
                  'Model: predicted new volume, interval set to zero, cm3': s_.pred_new_t0 / 1000,
                  'AUC within 15 mm: model': s_.auc_model_true, 'AUC within 15 mm: distance from baseline tumor': s_.auc_proximity,
                  'AUC within 15 mm: FLAIR intensity': s_.auc_intensity})
    if 'lesion_cf_dice' in r:
        o['Masks including resection cavity, no-change reference: Dice'] = r.lesion_cf_dice
        o['Masks including resection cavity, no-change reference: HD95, mm'] = r.lesion_cf_hd95
    rows.append(o)
T = pd.DataFrame(rows)
readme = pd.DataFrame({'Item': [
    'Dataset', 'Masks', 'No-change reference', 'Model', 'Label controls', 'Metrics', 'Volumes', 'Clinical data', 'Location and time', 'Missing values'],
    'Description': [
    'University of California San Francisco Adult Longitudinal Post-Treatment Diffuse Glioma (UCSF-ALPTDG) MRI dataset; '
    '112 patients, one baseline-follow-up pair each.',
    'Tumor masks comprise nonenhancing tumor core, surrounding nonenhancing FLAIR hyperintensity and enhancing tissue '
    '(labels 1-3); the resection cavity (label 4) is excluded unless stated otherwise.',
    'Baseline tumor mask carried forward unchanged as the prediction of the follow-up tumor extent.',
    'Conditioned three-dimensional U-Net (archived weights); input: baseline FLAIR image with N4 bias-field correction '
    'and z-score normalization within non-zero voxels; residual-tumor enforcement; probability threshold 0.5.',
    'Labels permuted: all three conditioning variables permuted across patients (fixed permutation, seed 42). Cortical '
    'architecture swapped: isocortical tumors assigned mesocortical, all other tumors assigned isocortical.',
    'Dice, Dice similarity coefficient; HD95, 95th-percentile Hausdorff distance; NSD, normalized surface Dice. '
    'Dice within new-growth region: overlap of predicted and observed tumor voxels outside the baseline mask.',
    'Volumes in cm3 at 1-mm isotropic resolution (1 voxel = 0.001 cm3). Interval from the dataset clinical table; the '
    'interval used for the time channel was derived from examination dates and was up to one day shorter.',
    'Treatment before baseline, annotated change between the examinations and progression documented within 365 days after '
    'baseline as provided in the clinical table of the dataset (Supplementary Methods); absence of documented progression does '
    'not exclude later progression.',
    'Predicted new volume: predicted tumor voxels outside the baseline mask at threshold 0.5, for the actual interval and with '
    'the time channel set to zero days. AUC within 15 mm: ranking of brain voxels outside the baseline tumor mask and within '
    '15 mm of it by the model probability, by distance from the baseline tumor and by FLAIR intensity, against follow-up tumor.',
    'Not applicable values are empty.']})
out = os.path.join(B, 'out_srep', 'Supplementary_Data_1.xlsx'); os.makedirs(os.path.dirname(out), exist_ok=True)
with pd.ExcelWriter(out, engine='openpyxl') as w:
    readme.to_excel(w, sheet_name='README', index=False)
    T.round(4).to_excel(w, sheet_name='Per-patient results', index=False)
    for ws in w.book.worksheets:
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60 if ws.title == 'README' else 28)
print(out, T.shape)
