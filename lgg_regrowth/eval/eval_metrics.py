import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / (denom + eps))


def voxel_volume_mm3_from_header(header) -> float:
    sx, sy, sz = header.get_zooms()[:3]
    return float(sx * sy * sz)


def volume_mm3(mask: np.ndarray, header) -> float:
    return float(mask.astype(np.uint8).sum()) * voxel_volume_mm3_from_header(header)


def _surface_points(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    struct = np.ones((3, 3, 3), dtype=bool)
    er = binary_erosion(mask, structure=struct, border_value=0)
    surface = mask ^ er
    return np.argwhere(surface).astype(np.float32)


def _spacing_zyx_mm(header) -> np.ndarray:
    sx, sy, sz = header.get_zooms()[:3]
    return np.array([sz, sy, sx], dtype=np.float32)


def hd95_mm(pred: np.ndarray, gt: np.ndarray, header, empty_value=0.0, one_empty_value=999.0) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return float(empty_value)
    if pred.sum() == 0 or gt.sum() == 0:
        return float(one_empty_value)

    A = _surface_points(pred)
    B = _surface_points(gt)
    spacing = _spacing_zyx_mm(header)
    A_mm = A * spacing
    B_mm = B * spacing

    tree_B = cKDTree(B_mm)
    tree_A = cKDTree(A_mm)
    dA, _ = tree_B.query(A_mm, k=1)
    dB, _ = tree_A.query(B_mm, k=1)
    dists = np.concatenate([dA, dB], axis=0)
    return float(np.percentile(dists, 95))


def nsd_mm(pred: np.ndarray, gt: np.ndarray, header, tol_mm: float, empty_value=1.0, one_empty_value=0.0) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return float(empty_value)
    if pred.sum() == 0 or gt.sum() == 0:
        return float(one_empty_value)

    A = _surface_points(pred)
    B = _surface_points(gt)
    if len(A) == 0 or len(B) == 0:
        return float(one_empty_value)

    spacing = _spacing_zyx_mm(header)
    A_mm = A * spacing
    B_mm = B * spacing

    tree_B = cKDTree(B_mm)
    tree_A = cKDTree(A_mm)

    dA, _ = tree_B.query(A_mm, k=1)
    dB, _ = tree_A.query(B_mm, k=1)

    within = (dA <= tol_mm).sum() + (dB <= tol_mm).sum()
    total = len(dA) + len(dB)
    return float(within / max(1, total))


def lesion_detect_case(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    p = pred.sum() > 0
    g = gt.sum() > 0
    if p and g:
        return 1, 0, 0
    if p and not g:
        return 0, 1, 0
    if (not p) and g:
        return 0, 0, 1
    return 0, 0, 0


def subsample_voxels_for_curves(prob: np.ndarray, gt: np.ndarray, max_samples: int, rng: np.random.Generator):
    p = prob.reshape(-1).astype(np.float32)
    y = gt.reshape(-1).astype(np.uint8)

    n = p.shape[0]
    if n <= max_samples:
        return p, y

    idx = rng.choice(n, size=max_samples, replace=False)
    return p[idx], y[idx]


def curves_for_probs(p_all: np.ndarray, y_all: np.ndarray, calibration_nbins: int):
    prec, rec, _ = precision_recall_curve(y_all, p_all)
    ap = average_precision_score(y_all, p_all)

    fpr, tpr, _ = roc_curve(y_all, p_all)
    roc_auc = auc(fpr, tpr)

    frac_pos, mean_pred = calibration_curve(y_all, p_all, n_bins=calibration_nbins, strategy="uniform")
    brier = brier_score_loss(y_all, p_all)

    bin_ids = np.minimum(
        calibration_nbins - 1,
        (p_all * calibration_nbins).astype(int)
    )
    ece = 0.0
    for b in range(calibration_nbins):
        m = (bin_ids == b)
        if not np.any(m):
            continue
        acc = float(y_all[m].mean())
        conf = float(p_all[m].mean())
        w = float(m.mean())
        ece += w * abs(acc - conf)

    return {
        "pr": (prec, rec, float(ap)),
        "roc": (fpr, tpr, float(roc_auc)),
        "calib": (frac_pos, mean_pred, float(ece), float(brier)),
        "ap": float(ap),
        "auc": float(roc_auc),
        "ece": float(ece),
        "brier": float(brier),
    }


def save_curves(out_dir: str, curves: Dict):
    os.makedirs(out_dir, exist_ok=True)

    prec, rec, ap = curves["pr"]
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.4f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), bbox_inches="tight", dpi=160)
    plt.close()

    fpr, tpr, roc_auc = curves["roc"]
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (AUC={roc_auc:.4f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), bbox_inches="tight", dpi=160)
    plt.close()

    frac_pos, mean_pred, ece, brier = curves["calib"]
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction positive")
    plt.title(f"Calibration (ECE={ece:.4f}, Brier={brier:.4f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "calibration_curve.png"), bbox_inches="tight", dpi=160)
    plt.close()


def save_volume_correlation_scatter(out_path: str, vol_gt: np.ndarray, vol_pred: np.ndarray, title: str):
    plt.figure()
    plt.scatter(vol_gt, vol_pred, s=10, alpha=0.7)
    mx = float(max(np.max(vol_gt), np.max(vol_pred))) if len(vol_gt) else 1.0
    plt.plot([0, mx], [0, mx], linestyle="--")
    if len(vol_gt) >= 2 and len(vol_pred) >= 2:
        try:
            r, _ = pearsonr(vol_gt, vol_pred)
            title = f"{title} (r={r:.4f})"
        except Exception:
            pass
    plt.xlabel("GT volume (mm^3)")
    plt.ylabel("Predicted volume (mm^3)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()