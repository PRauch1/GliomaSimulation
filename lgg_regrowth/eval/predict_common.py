import os
from typing import Callable, Dict, Optional

import numpy as np
import torch

import constants
from eval.eval_io import (
    build_time_channel,
    compute_global_t_max_days,
    find_meta_jsons,
    get_case_out_dir,
    pad_to_multiple,
    save_nifti,
    unpad_volume,
)

USE_AMP_INFER = True


@torch.no_grad()
def apply_model_baseline(
    model,
    device: torch.device,
    baseline: np.ndarray,
    residual: np.ndarray,
    target_days: float,
    t_max_days: float,
    enforce_residual: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    t_ch = build_time_channel(baseline, target_days, t_max_days)
    x = np.stack([baseline, residual, t_ch], axis=0).astype(np.float32)
    x_pad, pads = pad_to_multiple(x, int(constants.PAD_MULTIPLE))

    x_t = torch.from_numpy(x_pad[None, ...]).to(device)

    with torch.cuda.amp.autocast(enabled=(USE_AMP_INFER and device.type == "cuda")):
        logits = model(x_t)
        prob = torch.sigmoid(logits).float().cpu().numpy()

    prob_raw = unpad_volume(prob, pads).astype(np.float32)
    if enforce_residual:
        prob_enf = np.maximum(prob_raw, residual.astype(np.float32))
    else:
        prob_enf = prob_raw.copy()

    return prob_raw, prob_enf


@torch.no_grad()
def apply_model_film(
    model,
    device: torch.device,
    baseline: np.ndarray,
    residual: np.ndarray,
    meta_cat: np.ndarray,
    target_days: float,
    t_max_days: float,
    enforce_residual: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    t_ch = build_time_channel(baseline, target_days, t_max_days)
    x = np.stack([baseline, residual, t_ch], axis=0).astype(np.float32)
    x_pad, pads = pad_to_multiple(x, int(constants.PAD_MULTIPLE))

    x_t = torch.from_numpy(x_pad[None, ...]).to(device)
    meta_cat_t = torch.from_numpy(meta_cat[None, ...]).to(device).long()

    with torch.cuda.amp.autocast(enabled=(USE_AMP_INFER and device.type == "cuda")):
        logits = model(x_t, meta_cat_t)
        prob = torch.sigmoid(logits).float().cpu().numpy()

    prob_raw = unpad_volume(prob, pads).astype(np.float32)
    if enforce_residual:
        prob_enf = np.maximum(prob_raw, residual.astype(np.float32))
    else:
        prob_enf = prob_raw.copy()

    return prob_raw, prob_enf


def predict_cases_common(
    *,
    data_root: str,
    model_bundle: Dict,
    target_days: float,
    threshold: float,
    enforce_residual: bool,
    global_out_dir: Optional[str],
    save_predictions: bool,
    out_dir_name: str,
    build_case_context_fn: Callable,
    predict_one_fn: Callable,
    maybe_compute_dice_fn: Optional[Callable] = None,
) -> None:
    meta_paths = find_meta_jsons(data_root)
    t_max_from_ckpt = model_bundle[constants.CKPT_MAX_DAYS]
    t_max_days = t_max_from_ckpt if t_max_from_ckpt is not None else compute_global_t_max_days(meta_paths)
    model_bundle[constants.CKPT_MAX_DAYS] = t_max_days

    print("Using device:", model_bundle["device"])
    print(f"Found {len(meta_paths)} cases | t_max_days={t_max_days:.1f}")

    for meta_path in meta_paths:
        case_ctx = build_case_context_fn(meta_path, model_bundle)
        patient_id = case_ctx[constants.KEY_PATIENT_ID]
        op_id = case_ctx[constants.KEY_OP_ID]

        print(f"\n=== FORECAST [{patient_id} / {op_id}] @ {target_days} days ===")

        prob_raw, prob_enf = predict_one_fn(
            case_ctx["model"],
            case_ctx["device"],
            case_ctx,
            float(target_days),
            float(t_max_days),
            enforce_residual,
        )

        mask_raw = (prob_raw >= threshold).astype(np.uint8)
        mask_enf = (prob_enf >= threshold).astype(np.uint8)

        if save_predictions:
            out_dir = get_case_out_dir(
                case_ctx["preproc_root"],
                out_dir_name=out_dir_name,
                global_out_dir=global_out_dir,
            )
            day_i = int(target_days)

            save_nifti(
                os.path.join(out_dir, f"{constants.FILE_PRED_PROB_RAW}_{day_i}d.nii.gz"),
                prob_raw,
                case_ctx["affine"],
                case_ctx["header"],
                np.float32,
            )
            save_nifti(
                os.path.join(out_dir, f"{constants.FILE_PRED_PROB_ENF}_{day_i}d.nii.gz"),
                prob_enf,
                case_ctx["affine"],
                case_ctx["header"],
                np.float32,
            )
            save_nifti(
                os.path.join(out_dir, f"{constants.FILE_PRED_MASK_RAW}_{day_i}d_thr{threshold:.2f}.nii.gz"),
                mask_raw,
                case_ctx["affine"],
                case_ctx["header"],
                np.uint8,
            )
            save_nifti(
                os.path.join(out_dir, f"{constants.FILE_PRED_MASK_ENF}_{day_i}d_thr{threshold:.2f}.nii.gz"),
                mask_enf,
                case_ctx["affine"],
                case_ctx["header"],
                np.uint8,
            )
            print("  Saved forecast outputs to:", out_dir)

        if maybe_compute_dice_fn is not None:
            maybe_compute_dice_fn(
                case_ctx["preproc_root"],
                case_ctx["meta"],
                float(target_days),
                mask_enf,
            )