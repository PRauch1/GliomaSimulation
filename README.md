# Biologically Conditioned Deep Learning for Glioma Growth Prediction

This repository contains the training, evaluation, and inference code for the biologically conditioned 3D U-Net described in:

> **Biologically conditioned deep learning predicts glioma growth patterns beyond imaging alone**
> Wartha M, Böhm P, Atli B, Gmeiner M, Aichholzer M, Serra C, Sonnberger M, Stroh N, Aspalter S, Aufschnaiter-Hiessböck K, Rossmann T, Leibetseder A, Pichler J, Thomae W, Dorfer C, Gruber A, Stefanits H, Rauch P.

## Overview

Standard deep learning models for glioma growth prediction treat progression as geometric extrapolation from imaging alone. This framework conditions a 3D U-Net on three biological variables routinely available for diffuse lower-grade gliomas — **IDH mutation status**, **histological subtype**, and **cortical architecture type** — via Feature-wise Linear Modulation (FiLM). A residual tumour enforcement prior preserves confirmed postoperative disease in all predictions.

The model was trained on 159 patients (Kepler University Hospital, Linz) and externally validated on 112 independent patients from the publicly available [UCSF-ALPTDG dataset](https://doi.org/10.1148/ryai.230182).

## Repository Structure

```
lgg_regrowth/
├── constants.py              # Global constants, feature flags, file naming
├── train.py                  # Train imaging-only baseline (no FiLM)
├── train_film.py             # Train biologically conditioned model (FiLM)
├── validate.py               # Validate baseline on external cohort
├── validate_film.py          # Validate FiLM model on external cohort
├── predict.py                # Run inference (baseline) on new cases
├── predict_film.py           # Run inference (FiLM) on new cases
├── core/
│   ├── models.py             # ResizeConvUNet3D and ResizeConvUNet3D_FiLM architectures
│   ├── data.py               # Dataset class, metadata loading, vocabulary building
│   ├── metrics.py            # Volume consistency and residual inclusion losses
│   ├── train_loop.py         # Shared training loop with early stopping
│   └── config.py             # Argument parsing for training scripts
└── eval/
    ├── eval_models.py        # Checkpoint loading and model reconstruction
    ├── eval_io.py            # NIfTI I/O, preprocessing, case context building
    ├── eval_metadata.py      # FiLM metadata lookup during evaluation
    ├── eval_metrics.py       # Dice, HD95, NSD, volumetric error, PR/ROC/calibration
    ├── eval_types.py         # Dataclass for per-follow-up metric rows
    ├── predict_common.py     # Shared inference logic (baseline and FiLM)
    └── validate_common.py    # Full validation pipeline with aggregate reporting
```

## Requirements

Python 3.10+ with CUDA-capable GPU recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch (>=2.0), MONAI, nibabel, scipy, scikit-learn, pandas, matplotlib, numpy.

## Data Preparation

The code expects preprocessed NIfTI volumes organised as follows:

```
dataset/
├── film.csv                          # Biological metadata (see below)
├── Patient_001/
│   └── op_1/
│       └── preprocessed/
│           ├── meta.json             # Case metadata (baseline, follow-ups, timing)
│           ├── baseline_t2.nii.gz    # Skull-stripped, bias-corrected, z-scored T2/FLAIR
│           ├── residual_mask.nii.gz  # Binary postoperative residual tumour mask
│           └── followup_mask_*.nii.gz # Binary follow-up tumour masks
├── Patient_002/
│   └── ...
```

**Preprocessing pipeline** (applied to all scans before use):

1. Skull stripping — [HD-BET](https://github.com/MIC-DKFZ/HD-BET)
2. N4 bias field correction — [ANTsPy](https://github.com/ANTsX/ANTsPy) or [SimpleITK](https://simpleitk.org/)
3. Z-score intensity normalisation within brain mask
4. Resampling to a common isotropic voxel grid

**`film.csv`** — one row per patient, with columns:

| Column   | Description                        | Values                                                                                  |
|----------|------------------------------------|-----------------------------------------------------------------------------------------|
| `id`     | Patient identifier                 | Must match `Patient_*` directory names                                                  |
| `cortex` | Cortical architecture type         | `isocortical`, `mesocortical`, `allocortical`, `white_matter`, `multifocal`, `cerebellar`, `gliomatosis` |
| `idh`    | IDH mutation status                | `mutant`, `wildtype`, or empty (encoded as unknown)                                     |
| `histo`  | Histological subtype               | `astrocytoma`, `oligodendroglioma`, or empty (encoded as unknown)                       |

**`meta.json`** — per surgical event, structured as:

```json
{
  "baseline": {
    "image": "baseline_t2.nii.gz",
    "residual_mask": "residual_mask.nii.gz"
  },
  "followups": [
    {
      "image": "followup_t2_001.nii.gz",
      "mask": "followup_mask_001.nii.gz",
      "day": 182,
      "timestamp": "2021-06-15"
    }
  ]
}
```

## Training

**Biologically conditioned model (FiLM):**

```bash
cd lgg_regrowth
python train_film.py \
    --data_root /path/to/dataset \
    --out_dir checkpoints \
    --epochs 50 \
    --batch_size 1 \
    --lr 1e-4 \
    --augment \
    --lambda_res 0.05 \
    --early_stop_patience 7
```

**Imaging-only baseline:**

```bash
python train.py \
    --data_root /path/to/dataset \
    --out_dir checkpoints_baseline \
    --epochs 50 \
    --batch_size 1 \
    --lr 1e-4 \
    --augment \
    --lambda_res 0.05 \
    --early_stop_patience 7
```

Both scripts use identical training loops (combined Dice + BCE loss, Adam optimiser, optional `ReduceLROnPlateau` scheduler via `--use_scheduler`). Checkpoints are saved based on best validation loss (segmentation + volume consistency, excluding residual inclusion loss).

### Ablation experiments

To train single-variable or pairwise ablation models, edit the feature flags in `constants.py`:

```python
# Example: cortex-only model
FILM_CORTEX = True
FILM_IDH = False
FILM_HISTO = False
```

All seven configurations reported in the paper (3 single, 3 pairwise, 1 full) were trained this way under identical conditions.

## External Validation

Run the full validation pipeline on a held-out dataset (e.g. UCSF-ALPTDG):

```bash
python validate_film.py \
    --data-root /path/to/validation/dataset \
    --model-path checkpoints/best_unet_film.pth \
    --threshold 0.5
```

This produces:
- `per_followup_metrics.csv` — per-patient Dice, HD95, volumetric error, NSD@1mm, NSD@2mm (raw and enforced)
- `summary.json` — aggregate statistics
- PR, ROC, and calibration curves (PNG)
- Volume correlation scatter plots

For the baseline model:

```bash
python validate.py \
    --data-root /path/to/validation/dataset \
    --model-path checkpoints_baseline/best_unet_film.pth \
    --threshold 0.5
```

## Inference on New Cases

Generate a growth prediction for a specific target timepoint:

```bash
python predict_film.py \
    --data-root /path/to/patient/data \
    --model-path checkpoints/best_unet_film.pth \
    --target-days 180 \
    --threshold 0.5
```

Outputs per case:
- `pred_prob_raw_*.nii.gz` — voxel-wise tumour probability (raw network output)
- `pred_prob_enf_*.nii.gz` — probability with residual enforcement applied
- `pred_mask_raw_*.nii.gz` — binary prediction at specified threshold
- `pred_mask_enf_*.nii.gz` — binary prediction with enforcement at threshold

## Model Architecture

| Component              | Details                                                        |
|------------------------|----------------------------------------------------------------|
| Backbone               | 3D U-Net with 4 encoder levels (16→32→64→128 channels)        |
| Normalisation          | Instance normalisation (affine)                                |
| Activation             | LeakyReLU (slope 0.01)                                         |
| Upsampling             | Trilinear interpolation + 1×1×1 reduce conv                   |
| Conditioning           | FiLM layers at every encoder and decoder level                 |
| Meta encoder           | Per-variable learned embeddings (dim 16) → 2-layer MLP (dim 64) |
| FiLM initialisation    | Zero-initialised (identity transform at start of training)     |
| Input channels         | 3: T2/FLAIR, residual tumour mask, log-normalised time channel |
| Output                 | Sigmoid probability map (single channel)                       |
| Loss                   | Dice + BCE + λ_res × residual inclusion loss                   |
| Residual enforcement   | `max(prediction, baseline_mask)` applied at inference           |

## Evaluation Metrics

All metrics are computed per patient on the external validation cohort:

- **Dice coefficient** — volumetric overlap
- **HD95** — 95th percentile Hausdorff distance (mm), computed from surface points using spacing-aware cKDTree
- **Absolute volumetric error** — |predicted − ground truth| in mm³
- **Pearson volume correlation** — rank-order agreement of predicted vs. true volumes
- **NSD@τ** — normalised surface Dice at tolerance τ (1 mm and 2 mm), measuring boundary precision
- **PR/ROC AUC, ECE, Brier score** — voxel-level probabilistic calibration

## Citation

If you use this code, please cite:

```
[Citation will be added upon publication]
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
