import glob
import json
import os
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np

import constants
from eval.eval_metadata import meta_for_patient


def find_meta_jsons(data_root: str) -> List[str]:
    pattern = os.path.join(data_root, "Patient_*", "op_*", "preprocessed", constants.META_FILE_NAME)
    meta_paths = sorted(glob.glob(pattern))
    if not meta_paths:
        raise FileNotFoundError(f"No meta.json found under {pattern}")
    return meta_paths


def compute_global_t_max_days(meta_paths: List[str]) -> float:
    max_day = 1.0
    for mp in meta_paths:
        with open(mp) as f:
            meta = json.load(f)
        for fu in meta.get(constants.KEY_FOLLOWUPS, []):
            d = fu.get(constants.KEY_DAY, None)
            if d is None:
                continue
            try:
                max_day = max(max_day, float(d))
            except Exception:
                pass
    return float(max_day)


def zscore_normalize(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mask = vol != 0
    if mask.any():
        vals = vol[mask]
        mean = float(vals.mean())
        std = float(vals.std())
    else:
        mean = float(vol.mean())
        std = float(vol.std())

    std = std if std > eps else 1.0
    out = (vol - mean) / std
    return out.astype(np.float32)


def load_baseline_and_residual(meta_path: str):
    with open(meta_path) as f:
        meta = json.load(f)

    preproc_root = os.path.dirname(meta_path)

    bl_rel = meta[constants.KEY_BASELINE][constants.KEY_IMAGE]
    bl_path = os.path.join(preproc_root, bl_rel)

    bl_mask_rel = meta[constants.KEY_BASELINE].get(constants.KEY_BASELINE_MASK, None)
    bl_mask_path = os.path.join(preproc_root, bl_mask_rel) if bl_mask_rel else None

    if not os.path.exists(bl_path):
        raise FileNotFoundError(f"Baseline not found: {bl_path}")

    bl_img = nib.load(bl_path)
    baseline = zscore_normalize(bl_img.get_fdata().astype(np.float32))

    if bl_mask_path and os.path.exists(bl_mask_path):
        rm_img = nib.load(bl_mask_path)
        residual = (rm_img.get_fdata() > 0.5).astype(np.float32)
        if residual.shape != baseline.shape:
            raise ValueError(f"Residual shape {residual.shape} != baseline shape {baseline.shape}")
    else:
        residual = np.zeros_like(baseline, dtype=np.float32)

    return baseline, residual, bl_img.affine, bl_img.header, preproc_root, meta


def build_time_channel(baseline: np.ndarray, target_days: float, t_max_days: float) -> np.ndarray:
    target_days = 0.0 if target_days is None else float(target_days)
    t_max_days = max(1.0, float(t_max_days))
    t_norm = float(np.log1p(target_days) / np.log1p(t_max_days))
    return np.full_like(baseline, t_norm, dtype=np.float32)


def pad_to_multiple(x: np.ndarray, k: int):
    _, D, H, W = x.shape

    pad_D = (k - D % k) % k
    pad_H = (k - H % k) % k
    pad_W = (k - W % k) % k

    pb_D = pad_D // 2
    pa_D = pad_D - pb_D
    pb_H = pad_H // 2
    pa_H = pad_H - pb_H
    pb_W = pad_W // 2
    pa_W = pad_W - pb_W

    x_padded = np.pad(
        x,
        pad_width=((0, 0), (pb_D, pa_D), (pb_H, pa_H), (pb_W, pa_W)),
        mode="constant",
        constant_values=0.0,
    )
    pads = (pb_D, pa_D, pb_H, pa_H, pb_W, pa_W)
    return x_padded, pads


def unpad_volume(v: np.ndarray, pads):
    pb_D, pa_D, pb_H, pa_H, pb_W, pa_W = pads
    while v.ndim > 3:
        v = v[0]

    D, H, W = v.shape
    d0 = pb_D
    d1 = D - pa_D if pa_D > 0 else D
    h0 = pb_H
    h1 = H - pa_H if pa_H > 0 else H
    w0 = pb_W
    w1 = W - pa_W if pa_W > 0 else W
    return v[d0:d1, h0:h1, w0:w1]


def get_patient_op_ids(meta_path: str) -> Tuple[str, str]:
    op_dir = os.path.dirname(os.path.dirname(meta_path))
    patient_dir = os.path.dirname(op_dir)
    return os.path.basename(patient_dir), os.path.basename(op_dir)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_case_out_dir(preproc_root: str, out_dir_name: str, global_out_dir: str = None) -> str:
    if global_out_dir:
        return ensure_dir(global_out_dir)
    return ensure_dir(os.path.join(preproc_root, out_dir_name))


def save_nifti(path: str, arr: np.ndarray, affine, header, dtype) -> None:
    arr = arr.astype(dtype, copy=False)
    img = nib.Nifti1Image(arr, affine, header=header)
    nib.save(img, path)


def build_case_context_baseline(meta_path: str, model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    baseline, residual, affine, header, preproc_root, meta = load_baseline_and_residual(meta_path)
    patient_id, op_id = get_patient_op_ids(meta_path)

    # avoid circular import
    from eval.predict_common import apply_model_baseline

    return {
        "patient_id": patient_id,
        "op_id": op_id,
        "baseline": baseline,
        "residual": residual,
        "affine": affine,
        "header": header,
        "preproc_root": preproc_root,
        "meta": meta,
        "model": model_bundle[constants.CKPT_MODEL],
        "device": model_bundle[constants.CKPT_DEVICE],
        "t_max_days": model_bundle[constants.CKPT_MAX_DAYS],
        "predict_fn": apply_model_baseline,
    }

def build_case_context_film(meta_path: str, model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    baseline, residual, affine, header, preproc_root, meta = load_baseline_and_residual(meta_path)
    patient_id, op_id = get_patient_op_ids(meta_path)

    meta_bundle = model_bundle[constants.CKPT_META_BUNDLE]
    meta_cat = meta_for_patient(
        patient_id=patient_id,
        add_lookup=model_bundle[constants.CKPT_ADD_LOOKUP],
        cat_order=meta_bundle[constants.CKPT_CAT_ORDER],
        vocabs=meta_bundle[constants.CKPT_VOCABS],
    )

    # avoid circular import
    from eval.predict_common import apply_model_film

    return {
        "patient_id": patient_id,
        "op_id": op_id,
        "baseline": baseline,
        "residual": residual,
        "affine": affine,
        "header": header,
        "preproc_root": preproc_root,
        "meta": meta,
        "meta_cat": meta_cat,
        "model": model_bundle[constants.CKPT_MODEL],
        "device": model_bundle[constants.CKPT_DEVICE],
        "t_max_days": model_bundle[constants.CKPT_MAX_DAYS],
        "predict_fn": apply_model_film,
    }