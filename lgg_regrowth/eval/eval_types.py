from dataclasses import dataclass
from typing import Dict


@dataclass
class RowMetrics:
    patient_id: str
    op_id: str
    fu_ts: str
    fu_day: float

    # RAW
    dice_raw: float
    hd95_raw: float
    vol_pred_raw: float
    vol_err_abs_raw: float

    # ENF
    dice_enf: float
    hd95_enf: float
    vol_pred_enf: float
    vol_err_abs_enf: float

    # GT
    vol_gt: float

    # NSD at tolerances
    nsd_raw: Dict[float, float]
    nsd_enf: Dict[float, float]

    # detection flags (enforced)
    det_tp: int
    det_fp: int
    det_fn: int