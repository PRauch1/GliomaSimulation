# External validation on UCSF-ALPTDG

Scripts for the external validation of the conditioned growth model (`lgg_regrowth`) on the UCSF-ALPTDG dataset
against a no-change reference (tumor masks without resection cavity), as reported in:

> Toward digital twins of postoperative glioma growth with a three-dimensional U-Net informed by cortical
> architecture, isocitrate dehydrogenase status and histology.

The same files are provided as Supplementary Software 1 of the manuscript and in the folder `validation_ucsf_alptdg/`
of https://github.com/PRauch1/GliomaSimulation (model code in `lgg_regrowth/`, archived at
https://doi.org/10.5281/zenodo.19268239).

## Expected layout

Paths are relative to this folder. The scripts locate the model code in `../lgg_regrowth`; set `LGG_REPO` to use
another location.

```
infer/model.safetensors, model_config.json, preprocessing.json   archived weights and configuration
data/ucsf/raw/<id>/<id>_time1_flair.nii.gz, <id>_time1_t2.nii.gz, <id>_time1_seg.nii.gz, <id>_time2_seg.nii.gz
data/ucsf/files.csv                                               id, t2_bl, seg_bl, seg_fu (paths)
data/new/feat_val.xlsx                                            conditioning variables of the validation cohort
data/nc/per_followup_metrics_0.5_idh_histo_cortex.csv            per-patient results of the earlier evaluation
                                                                  (interval used for the time channel; agreement check)
data/ucsf/raw/UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx   clinical table distributed with the dataset
                                                                  (treatment, progression, annotated change)
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/ucsf_reference_metrics.py` | No-change reference (baseline mask carried forward) for lesion and tumor masks |
| `scripts/repro_probe2.py` | Comparison of preprocessing variants with the earlier per-patient evaluation |
| `scripts/ucsf_infer.py` | Inference (FLAIR or T2 input, optional N4), label controls, threshold sweep, cavity analysis, calibration counts |
| `analysis_tumor.py` | Summary statistics, paired comparisons with the no-change reference, bootstrap confidence intervals |
| `clinical_srep.py` | Treatment before baseline, annotated change, documented progression, interval dependence, criterion for improving on no change |
| `scripts/ucsf_spatial.py` | Ranking of brain voxels outside the baseline tumor mask and within 15 mm of it (model, distance, FLAIR intensity), oracle- and matched-volume placement, time probe (interval set to zero) |
| `spatial_srep.py` | Summary of the spatial analysis and time probe |
| `subgroups_srep.py` | Descriptive subgroup results |
| `numbers_srep.py` | Collection of all reported values; also spatial ranking by annotated change, interval by annotated increase and tumors without histomorphologic specification |
| `figs_srep.py`, `fig3_select.py`, `fig3_srep.py`, `fig4_srep.py` | Figures |
| `make_suppdata.py` | Supplementary Data 1 |

## Primary run

Run from this folder.

```
python scripts/ucsf_infer.py --repo ../lgg_regrowth --weights infer/model.safetensors --config infer/model_config.json \
  --files data/ucsf/files.csv --features data/new/feat_val.xlsx \
  --days data/nc/per_followup_metrics_0.5_idh_histo_cortex.csv --out data/ucsf/infer_flair \
  --masks tumor --variants true,all_perm,cortex_swap --modality flair --n4 --extra-lesion
python scripts/ucsf_spatial.py --repo ../lgg_regrowth --weights infer/model.safetensors --config infer/model_config.json \
  --files data/ucsf/files.csv --features data/new/feat_val.xlsx \
  --days data/nc/per_followup_metrics_0.5_idh_histo_cortex.csv --out data/ucsf/spatial_flair
python analysis_tumor.py && python clinical_srep.py && python subgroups_srep.py && python spatial_srep.py && python numbers_srep.py
```

Figure 3: `python fig3_select.py` lists the patients selected by the rules given in the legend;
`python fig3_srep.py 100158,100147,100036` draws the figure with the preprocessing of the primary run (FLAIR input,
N4 bias-field correction).

Sensitivity run with T2 input: `--modality t2 --flip-axes 0` (without `--n4`), analysed with
`UCSF_RUN=<dir> UCSF_OUT=tumor_results_t2.json python analysis_tumor.py`.

Dependencies: Python 3, PyTorch, NumPy, SciPy, pandas, scikit-learn, nibabel, SimpleITK, safetensors, matplotlib.
