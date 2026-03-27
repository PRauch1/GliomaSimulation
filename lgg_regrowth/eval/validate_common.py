import json
import os
from dataclasses import asdict
from typing import Callable, Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd

import constants
from eval.eval_io import (
    compute_global_t_max_days,
    ensure_dir,
    find_meta_jsons,
    get_case_out_dir,
    save_nifti,
)
from eval.eval_metrics import (
    curves_for_probs,
    dice_score,
    hd95_mm,
    lesion_detect_case,
    nsd_mm,
    save_curves,
    save_volume_correlation_scatter,
    subsample_voxels_for_curves,
    volume_mm3,
)
from eval.eval_types import RowMetrics

def maybe_compute_dice_at_target_day(preproc_root: str, meta: Dict, target_days: float, pred_mask: np.ndarray):
    fus = meta.get(constants.KEY_FOLLOWUPS, [])
    if not fus:
        return

    best_fu = None
    best_diff = None
    for fu in fus:
        d = fu.get(constants.KEY_DAY, None)
        if d is None:
            continue
        diff = abs(float(d) - float(target_days))
        if best_diff is None or diff < best_diff:
            best_fu = fu
            best_diff = diff

    if best_fu is None:
        return

    ts = best_fu.get(constants.KEY_TS, "")
    fu_mask_rel = best_fu.get(constants.KEY_MASK, None)
    if fu_mask_rel is None:
        return

    fu_mask_path = os.path.join(preproc_root, fu_mask_rel)
    if not os.path.exists(fu_mask_path):
        return

    fu_img = nib.load(fu_mask_path)
    fu_mask = (fu_img.get_fdata() > 0.5).astype(np.uint8)

    if fu_mask.shape != pred_mask.shape:
        return

    d = dice_score(pred_mask, fu_mask)
    print(f"  Closest FU Dice at ~{target_days}d (ts={ts}, diff={best_diff:.1f}d): {d:.4f}")


def run_validation_case(
    *,
    case_ctx: Dict,
    threshold: float,
    max_vox_samples_per_fu: int,
    nsd_tols_mm,
    min_mask_voxels: int,
    save_predictions: bool,
    rng: np.random.Generator,
    predict_one_fn: Callable,
) -> Dict:
    patient_id = case_ctx[constants.KEY_PATIENT_ID]
    op_id = case_ctx[constants.KEY_OP_ID]
    baseline = case_ctx["baseline"]
    residual = case_ctx["residual"]
    affine = case_ctx["affine"]
    header_bl = case_ctx["header"]
    preproc_root = case_ctx["preproc_root"]
    meta = case_ctx["meta"]
    model = case_ctx["model"]
    device = case_ctx["device"]
    t_max_days = case_ctx["t_max_days"]
    out_dir = case_ctx["out_dir"]

    residual_voxels = int(residual.sum())

    fus = meta.get(constants.KEY_FOLLOWUPS, [])
    if not fus:
        print("  (no follow-ups, skipping)")
        return {"rows": [], "curve_p_raw": [], "curve_y": [], "curve_p_enf": [], "curve_y2": []}

    if residual_voxels < min_mask_voxels:
        print(f"  (residual mask too small: {residual_voxels} voxels < {min_mask_voxels}, skipping all follow-ups)")
        return {"rows": [], "curve_p_raw": [], "curve_y": [], "curve_p_enf": [], "curve_y2": []}

    rows: List[RowMetrics] = []
    curve_p_raw, curve_y = [], []
    curve_p_enf, curve_y2 = [], []

    for fu in fus:
        day = fu.get(constants.KEY_DAY, None)
        ts = fu.get(constants.KEY_TS, "")
        if day is None:
            print(f"  - FU ts={ts}: no day value, skipping")
            continue

        fu_mask_rel = fu[constants.KEY_MASK]
        fu_mask_path = os.path.join(preproc_root, fu_mask_rel)
        if not os.path.exists(fu_mask_path):
            print(f"  - FU ts={ts}, day={day}: mask missing, skipping")
            continue

        fu_img = nib.load(fu_mask_path)
        fu_mask = (fu_img.get_fdata() > 0.5).astype(np.uint8)

        if fu_mask.shape != baseline.shape:
            print(f"  - FU ts={ts}, day={day}: shape mismatch FU {fu_mask.shape} vs BL {baseline.shape}, skipping")
            continue

        fu_voxels = int(fu_mask.sum())
        if fu_voxels < min_mask_voxels:
            print(f"  - FU ts={ts}, day={day}: GT mask too small ({fu_voxels} voxels < {min_mask_voxels}), skipping")
            continue

        prob_raw, prob_enf = predict_one_fn(
            model,
            device,
            case_ctx,
            float(day),
            float(t_max_days),
            True,
        )

        pred_raw = (prob_raw >= threshold).astype(np.uint8)
        pred_enf = (prob_enf >= threshold).astype(np.uint8)

        d_raw = dice_score(pred_raw, fu_mask)
        d_enf = dice_score(pred_enf, fu_mask)

        hd_raw = hd95_mm(pred_raw, fu_mask, fu_img.header)
        hd_enf = hd95_mm(pred_enf, fu_mask, fu_img.header)

        vol_g = volume_mm3(fu_mask, fu_img.header)
        vol_p_raw = volume_mm3(pred_raw, fu_img.header)
        vol_p_enf = volume_mm3(pred_enf, fu_img.header)

        err_raw = abs(vol_p_raw - vol_g)
        err_enf = abs(vol_p_enf - vol_g)

        nsd_raw_map = {tol: nsd_mm(pred_raw, fu_mask, fu_img.header, tol) for tol in nsd_tols_mm}
        nsd_enf_map = {tol: nsd_mm(pred_enf, fu_mask, fu_img.header, tol) for tol in nsd_tols_mm}

        tp, fp, fn = lesion_detect_case(pred_enf, fu_mask)

        print(
            f"  - FU ts={ts}, day={day} | "
            f"Dice raw={d_raw:.4f}, enf={d_enf:.4f} | "
            f"HD95 raw={hd_raw:.2f}mm, enf={hd_enf:.2f}mm | "
            f"Vol GT={vol_g:.1f}mm3 raw={vol_p_raw:.1f} enf={vol_p_enf:.1f} | "
            + " ".join([f"NSD@{tol:.0f} raw={nsd_raw_map[tol]:.3f} enf={nsd_enf_map[tol]:.3f}" for tol in nsd_tols_mm])
        )

        rows.append(
            RowMetrics(
                patient_id=patient_id,
                op_id=op_id,
                fu_ts=str(ts),
                fu_day=float(day),
                dice_raw=float(d_raw),
                hd95_raw=float(hd_raw),
                vol_pred_raw=float(vol_p_raw),
                vol_err_abs_raw=float(err_raw),
                dice_enf=float(d_enf),
                hd95_enf=float(hd_enf),
                vol_pred_enf=float(vol_p_enf),
                vol_err_abs_enf=float(err_enf),
                vol_gt=float(vol_g),
                nsd_raw=nsd_raw_map,
                nsd_enf=nsd_enf_map,
                det_tp=int(tp),
                det_fp=int(fp),
                det_fn=int(fn),
            )
        )

        if save_predictions:
            save_nifti(os.path.join(out_dir, f"{constants.FILE_PRED_PROB_RAW}_fu_{ts}_d{int(float(day))}.nii.gz"), prob_raw, affine, header_bl, np.float32)
            save_nifti(os.path.join(out_dir, f"{constants.FILE_PRED_PROB_ENF}_fu_{ts}_d{int(float(day))}.nii.gz"), prob_enf, affine, header_bl, np.float32)
            save_nifti(os.path.join(out_dir, f"{constants.FILE_PRED_MASK_RAW}_fu_{ts}_d{int(float(day))}_thr{threshold:.2f}.nii.gz"), pred_raw, affine, header_bl, np.uint8)
            save_nifti(os.path.join(out_dir, f"{constants.FILE_PRED_MASK_ENF}_fu_{ts}_d{int(float(day))}_thr{threshold:.2f}.nii.gz"), pred_enf, affine, header_bl, np.uint8)

        p_raw_s, y_s = subsample_voxels_for_curves(prob_raw, fu_mask, max_vox_samples_per_fu, rng)
        p_enf_s, y_s2 = subsample_voxels_for_curves(prob_enf, fu_mask, max_vox_samples_per_fu, rng)

        curve_p_raw.append(p_raw_s)
        curve_y.append(y_s)
        curve_p_enf.append(p_enf_s)
        curve_y2.append(y_s2)

    return {
        "rows": rows,
        "curve_p_raw": curve_p_raw,
        "curve_y": curve_y,
        "curve_p_enf": curve_p_enf,
        "curve_y2": curve_y2,
    }


def _rows_to_dataframe(rows: List[RowMetrics], nsd_tols_mm) -> pd.DataFrame:
    records = []
    for r in rows:
        rec = {
            "patient_id": r.patient_id,
            "op_id": r.op_id,
            "fu_ts": r.fu_ts,
            "fu_day": r.fu_day,
            "dice_raw": r.dice_raw,
            "hd95_raw": r.hd95_raw,
            "vol_pred_raw": r.vol_pred_raw,
            "vol_err_abs_raw": r.vol_err_abs_raw,
            "dice_enf": r.dice_enf,
            "hd95_enf": r.hd95_enf,
            "vol_pred_enf": r.vol_pred_enf,
            "vol_err_abs_enf": r.vol_err_abs_enf,
            "vol_gt": r.vol_gt,
            "det_tp": r.det_tp,
            "det_fp": r.det_fp,
            "det_fn": r.det_fn,
        }
        for tol in nsd_tols_mm:
            rec[f"nsd_raw_{tol:g}mm"] = r.nsd_raw[tol]
            rec[f"nsd_enf_{tol:g}mm"] = r.nsd_enf[tol]
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _save_curve_artifacts(
    out_dir_global: str,
    df: pd.DataFrame,
    curve_p_raw: List[np.ndarray],
    curve_y: List[np.ndarray],
    curve_p_enf: List[np.ndarray],
    calibration_nbins: int,
    save_curves_flag: bool,
    curve_stats_filename: str,
):
    if len(curve_y) == 0:
        print("No voxel samples for curves; skipping curves.")
        return

    y_all = np.concatenate(curve_y, axis=0)
    p_raw_all = np.concatenate(curve_p_raw, axis=0)
    p_enf_all = np.concatenate(curve_p_enf, axis=0)

    if y_all.size == 0:
        print("No voxel samples for curves; skipping curves.")
        return

    curves_raw = curves_for_probs(p_raw_all, y_all, calibration_nbins)
    curves_enf = curves_for_probs(p_enf_all, y_all, calibration_nbins)

    out_curves_raw = ensure_dir(os.path.join(out_dir_global, constants.DIR_CURVES_RAW))
    out_curves_enf = ensure_dir(os.path.join(out_dir_global, constants.DIR_CURVES_ENF))

    if save_curves_flag:
        save_curves(out_curves_raw, curves_raw)
        save_curves(out_curves_enf, curves_enf)

    with open(os.path.join(out_curves_raw, curve_stats_filename), "w") as f:
        json.dump({k: curves_raw[k] for k in ("ap", "auc", "ece", "brier")}, f, indent=2)
    with open(os.path.join(out_curves_enf, curve_stats_filename), "w") as f:
        json.dump({k: curves_enf[k] for k in ("ap", "auc", "ece", "brier")}, f, indent=2)

    save_volume_correlation_scatter(
        os.path.join(out_dir_global, "volume_scatter_raw.png"),
        df["vol_gt"].to_numpy(),
        df["vol_pred_raw"].to_numpy(),
        "Volume correlation RAW",
    )
    save_volume_correlation_scatter(
        os.path.join(out_dir_global, "volume_scatter_enf.png"),
        df["vol_gt"].to_numpy(),
        df["vol_pred_enf"].to_numpy(),
        "Volume correlation ENF",
    )

    print("\n=== CURVE STATS (voxel-level) ===")
    print(f"RAW: AP={curves_raw['ap']:.4f} AUC={curves_raw['auc']:.4f} ECE={curves_raw['ece']:.4f} Brier={curves_raw['brier']:.4f}")
    print(f"ENF: AP={curves_enf['ap']:.4f} AUC={curves_enf['auc']:.4f} ECE={curves_enf['ece']:.4f} Brier={curves_enf['brier']:.4f}")
    print("Saved curve plots under:", out_dir_global)


def full_validation_common(
    *,
    data_root: str,
    model_bundle: Dict,
    threshold: float,
    max_vox_samples_per_fu: int,
    global_out_dir: Optional[str],
    save_predictions: bool,
    save_curves: bool,
    out_dir_name: str,
    nsd_tols_mm=(1.0, 2.0),
    calibration_nbins: int = 15,
    min_mask_voxels: int = 10,
    build_case_context_fn: Callable = None,
    predict_one_fn: Callable = None,
    curve_stats_filename: str = "curve_stats.json",
) -> None:
    meta_paths = find_meta_jsons(data_root)
    t_max_from_ckpt = model_bundle[constants.CKPT_MAX_DAYS]
    t_max_days = t_max_from_ckpt if t_max_from_ckpt is not None else compute_global_t_max_days(meta_paths)
    model_bundle[constants.CKPT_MAX_DAYS] = t_max_days

    print("Using device:", model_bundle[constants.CKPT_DEVICE])
    print(f"Found {len(meta_paths)} cases | t_max_days={t_max_days:.1f}")

    rng = np.random.default_rng(42)

    rows: List[RowMetrics] = []
    curve_p_raw, curve_y = [], []
    curve_p_enf, curve_y2 = [], []

    for meta_path in meta_paths:
        case_ctx = build_case_context_fn(meta_path, model_bundle)
        case_ctx["out_dir"] = get_case_out_dir(case_ctx["preproc_root"], out_dir_name=out_dir_name, global_out_dir=None)

        print(f"\n=== VALIDATION [{case_ctx['patient_id']} / {case_ctx['op_id']}] ===")
        case_res = run_validation_case(
            case_ctx=case_ctx,
            threshold=threshold,
            max_vox_samples_per_fu=max_vox_samples_per_fu,
            nsd_tols_mm=nsd_tols_mm,
            min_mask_voxels=min_mask_voxels,
            save_predictions=save_predictions,
            rng=rng,
            predict_one_fn=predict_one_fn,
        )
        rows.extend(case_res["rows"])
        curve_p_raw.extend(case_res["curve_p_raw"])
        curve_y.extend(case_res["curve_y"])
        curve_p_enf.extend(case_res["curve_p_enf"])
        curve_y2.extend(case_res["curve_y2"])

    if not rows:
        print("\nNo follow-up metrics computed.")
        return

    out_dir_global = global_out_dir or ensure_dir(os.path.join(data_root, constants.DIR_GLOBAL_EVAL))
    ensure_dir(out_dir_global)

    df = _rows_to_dataframe(rows, nsd_tols_mm)
    df.to_csv(os.path.join(out_dir_global, "per_followup_metrics.csv"), index=False)

    summary = {
        "n_followups": int(len(df)),
        "dice_raw_mean": float(df["dice_raw"].mean()),
        "dice_enf_mean": float(df["dice_enf"].mean()),
        "hd95_raw_mean": float(df["hd95_raw"].mean()),
        "hd95_enf_mean": float(df["hd95_enf"].mean()),
        "vol_err_abs_raw_mean": float(df["vol_err_abs_raw"].mean()),
        "vol_err_abs_enf_mean": float(df["vol_err_abs_enf"].mean()),
        "det_tp": int(df["det_tp"].sum()),
        "det_fp": int(df["det_fp"].sum()),
        "det_fn": int(df["det_fn"].sum()),
    }

    tp = summary["det_tp"]
    fp = summary["det_fp"]
    fn = summary["det_fn"]
    summary["det_sensitivity"] = float(tp / max(1, tp + fn))
    summary["det_precision"] = float(tp / max(1, tp + fp))
    denom = max(1e-12, 2 * tp + fp + fn)
    summary["det_f1"] = float((2 * tp) / denom)

    with open(os.path.join(out_dir_global, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    _save_curve_artifacts(
        out_dir_global=out_dir_global,
        df=df,
        curve_p_raw=curve_p_raw,
        curve_y=curve_y,
        curve_p_enf=curve_p_enf,
        calibration_nbins=calibration_nbins,
        save_curves_flag=save_curves,
        curve_stats_filename=curve_stats_filename,
    )