import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import constants

def _clean_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def _clean_int(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def load_additional_info(data_root: str, required_cols: List[str]) -> pd.DataFrame:
    csv_path = os.path.join(data_root, constants.FILE_FILM_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {constants.FILE_FILM_CSV} in data_root: {csv_path}")

    add_df = pd.read_csv(csv_path)

    expected = set(["id"] + list(required_cols))
    missing_cols = expected - set(add_df.columns)
    if missing_cols:
        raise ValueError(f"{constants.FILE_FILM_CSV} missing columns: {sorted(missing_cols)}")

    add_df["id"] = add_df["id"].astype(str).str.strip()

    for col in [constants.FILM_KEY_CORTEX, constants.FILM_KEY_IDH, constants.FILM_KEY_HISTO]:
        if col in add_df.columns:
            add_df[col] = add_df[col].apply(_clean_str)
            add_df.loc[add_df[col] == "", col] = constants.FILM_KEY_UNK
            add_df.loc[add_df[col].isna(), col] = constants.FILM_KEY_UNK

    return add_df


def make_addinfo_lookup(add_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    cols = [c for c in add_df.columns if c != "id"]
    for _, r in add_df.iterrows():
        pid = str(r["id"]).strip()
        lookup[pid] = {c: r.get(c, None) for c in cols}
    return lookup


def map_to_index_str(value, vocab: Dict[str, int]) -> int:
    v = _clean_str(value)
    if v == "":
        v = constants.FILM_KEY_UNK
    return int(vocab.get(v, vocab.get(constants.FILM_KEY_UNK, 0)))


def map_to_index_int(value, vocab: Dict[str, int]) -> int:
    v = _clean_int(value)
    if v is None:
        return int(vocab.get(constants.FILM_KEY_UNK, 0))
    return int(vocab.get(str(int(v)), vocab.get(constants.FILM_KEY_UNK, 0)))


def meta_for_patient(
    patient_id: str,
    add_lookup: Dict[str, Dict[str, Any]],
    cat_order: List[str],
    vocabs: Dict[str, Dict[str, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    info = add_lookup.get(patient_id, {})

    cat_idx: List[int] = []
    for k in cat_order:
        vocab = vocabs[k]
        if k in (constants.FILM_KEY_CORTEX, constants.FILM_KEY_IDH, constants.FILM_KEY_HISTO):
            cat_idx.append(map_to_index_str(info.get(k, constants.FILM_KEY_UNK), vocab))
        else:
            raise RuntimeError(f"Unexpected categorical key in cat_order: {k}")

    meta_cat = np.array(cat_idx, dtype=np.int64) if len(cat_idx) else np.zeros((0,), dtype=np.int64)
    return meta_cat


def print_distinct_for_check(add_df: pd.DataFrame, cat_order: List[str]):
    for k in cat_order:
        if k not in add_df.columns:
            print(f"[CHECK] Distinct {k}: column not found in CSV")
            continue

        vals = sorted(set(add_df[k].fillna(constants.FILM_KEY_UNK).astype(str).tolist()))
        print(f"[CHECK] Distinct {k} ({len(vals)}): {vals}")