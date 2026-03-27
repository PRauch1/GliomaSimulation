import os
import json
import glob
from typing import Tuple, Optional, Dict, List, Any

import numpy as np
import pandas as pd
import nibabel as nib

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import GroupShuffleSplit

# for mask dilation augmentation
try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None
    print("[WARN] scipy not found – mask dilation augmentation will be disabled.")

import constants

# ==========================
# Metadata CSV utilities
# ==========================
ADDINFO_CSV = "film.csv"

KEY_CORTEX = "cortex"
KEY_IDH = "idh"
KEY_HISTO = "histo"

UNK = "__UNK__"

# ==========================
# Basic sample dataframe
# ==========================
def get_df(data_root: str) -> pd.DataFrame:
    rows = []
    pattern = os.path.join(data_root, "Patient_*", "op_*", "preprocessed", constants.META_FILE_NAME)

    for meta_path in glob.glob(pattern):
        with open(meta_path) as f:
            meta = json.load(f)

        preproc_dir = os.path.dirname(meta_path)
        op_dir = os.path.dirname(preproc_dir)
        patient_dir = os.path.dirname(op_dir)

        patient_id = os.path.basename(patient_dir)
        root = preproc_dir

        baseline_img = os.path.join(root, meta[constants.KEY_BASELINE][constants.KEY_IMAGE])
        residual_mask = meta[constants.KEY_BASELINE][constants.KEY_BASELINE_MASK]
        residual_mask = os.path.join(root, residual_mask) if residual_mask else None

        for fup in meta[constants.KEY_FOLLOWUPS]:
            rows.append({
                constants.KEY_PATIENT_ID: patient_id,
                constants.KEY_ROOT: root,
                constants.KEY_BASELINE_IMG: baseline_img,
                constants.KEY_BASELINE_MASK: residual_mask,
                constants.KEY_FOLLOWUPS_IMG: os.path.join(root, fup[constants.KEY_IMAGE]),
                constants.KEY_FOLLOWUPS_MASK: os.path.join(root, fup[constants.KEY_MASK]),
                constants.KEY_DAY: fup[constants.KEY_DAY],
                constants.KEY_TS: fup[constants.KEY_TS],
            })

    return pd.DataFrame(rows)


def get_train_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """group-wise train/val split by patient_id"""
    groups = df[constants.KEY_PATIENT_ID]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

# ==========================
# FiLM metadata helpers
# ==========================
def clean_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def clean_int(x) -> Optional[int]:
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


def enabled_feature_keys() -> Dict[str, bool]:
    return {
        KEY_CORTEX: bool(getattr(constants, "FILM_CORTEX", False)),
        KEY_IDH: bool(getattr(constants, "FILM_IDH", False)),
        KEY_HISTO: bool(getattr(constants, "FILM_HISTO", False)),
    }


def load_additional_info(data_root: str) -> pd.DataFrame:
    csv_path = os.path.join(data_root, ADDINFO_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {ADDINFO_CSV} in data_root: {csv_path}")

    add_df = pd.read_csv(csv_path)

    enabled = enabled_feature_keys()
    required = {"id"} | {k for k, on in enabled.items() if on}
    missing_cols = required - set(add_df.columns)
    if missing_cols:
        raise ValueError(f"{ADDINFO_CSV} missing columns required by enabled features: {sorted(missing_cols)}")

    add_df["id"] = add_df["id"].astype(str).str.strip()

    # Clean categorical string fields
    for col in [KEY_CORTEX, KEY_IDH, KEY_HISTO]:
        if col in add_df.columns:
            add_df[col] = add_df[col].apply(clean_str)
            add_df.loc[add_df[col] == "", col] = UNK
            add_df.loc[add_df[col].isna(), col] = UNK

    return add_df

def make_addinfo_lookup(add_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    enabled = enabled_feature_keys()
    keys = [k for k, on in enabled.items() if on]

    lookup: Dict[str, Dict[str, Any]] = {}
    for _, r in add_df.iterrows():
        pid = str(r["id"]).strip()
        rec: Dict[str, Any] = {}
        for k in keys:
            rec[k] = r.get(k, None)
        lookup[pid] = rec
    return lookup


def build_vocab(values: List[str]) -> Dict[str, int]:
    uniq = set([clean_str(v) for v in values if clean_str(v) != ""])
    uniq.discard(UNK)
    vocab = {UNK: 0}
    for i, v in enumerate(sorted(uniq), start=1):
        vocab[v] = i
    return vocab

def build_vocab_int(values: List[Optional[int]]) -> Dict[str, int]:
    uniq_int = sorted({v for v in values if v is not None})
    vocab = {UNK: 0}
    for i, v in enumerate(uniq_int, start=1):
        vocab[str(int(v))] = i
    return vocab

def map_to_index_str(value: Any, vocab: Dict[str, int]) -> int:
    v = clean_str(value)
    if v == "":
        v = UNK
    return int(vocab.get(v, vocab.get(UNK, 0)))


def map_to_index_int(value: Any, vocab: Dict[str, int]) -> int:
    v = clean_int(value)
    if v is None:
        return int(vocab.get(UNK, 0))
    return int(vocab.get(str(int(v)), vocab.get(UNK, 0)))


def print_distinct_values(add_df: pd.DataFrame):
    enabled = enabled_feature_keys()

    if enabled.get(KEY_CORTEX, False) and KEY_CORTEX in add_df.columns:
        vals = sorted(set(add_df[KEY_CORTEX].fillna(UNK).astype(str).tolist()))
        print(f"[CHECK] Distinct {KEY_CORTEX} ({len(vals)}): {vals}")

    if enabled.get(KEY_IDH, False) and KEY_IDH in add_df.columns:
        vals = sorted(set(add_df[KEY_IDH].fillna(UNK).astype(str).tolist()))
        print(f"[CHECK] Distinct {KEY_IDH} ({len(vals)}): {vals}")

    if enabled.get(KEY_HISTO, False) and KEY_HISTO in add_df.columns:
        vals = sorted(set(add_df[KEY_HISTO].fillna(UNK).astype(str).tolist()))
        print(f"[CHECK] Distinct {KEY_HISTO} ({len(vals)}): {vals}")

def compute_t_max_days(train_df: pd.DataFrame, default: float = 365.0) -> float:
    if len(train_df) == 0:
        return float(default)

    day_series = pd.to_numeric(train_df[constants.KEY_DAY], errors="coerce")
    max_day = day_series.max(skipna=True)

    if pd.isna(max_day):
        return float(default)

    return max(1.0, float(max_day))

# ==========================
# Dataset
# ==========================
class TumourRegrowthDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        augment: bool = False,
        t_max_days: Optional[float] = None,
        add_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        vocabs: Optional[Dict[str, Dict[str, int]]] = None,
        enabled: Optional[Dict[str, bool]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.add_lookup = add_lookup or {}
        self.vocabs = vocabs or {}
        self.enabled = enabled or {}
        self.augment = augment
        self.pad_multiple = constants.PAD_MULTIPLE

        # define feature order (must match model + checkpoint)
        self.cat_order: List[str] = []

        # categorical strings
        for k in [KEY_CORTEX, KEY_IDH, KEY_HISTO]:
            if self.enabled.get(k, False):
                self.cat_order.append(k)

        self.use_film = len(self.cat_order) > 0

        if t_max_days is None:
            self.t_max_days = compute_t_max_days(self.df)
        else:
            self.t_max_days = max(1.0, float(t_max_days))

    def __len__(self):
        return len(self.df)

    def _load_nii(self, path: str) -> np.ndarray:
        img = nib.load(path)
        return img.get_fdata().astype(np.float32)

    @staticmethod
    def _zscore_normalize(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
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

    def _pad_to_multiple(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, D, H, W = x.shape
        k = self.pad_multiple

        pad_D = (k - D % k) % k
        pad_H = (k - H % k) % k
        pad_W = (k - W % k) % k

        bd = pad_D // 2; ad = pad_D - bd
        bh = pad_H // 2; ah = pad_H - bh
        bw = pad_W // 2; aw = pad_W - bw

        x = np.pad(x, ((0, 0), (bd, ad), (bh, ah), (bw, aw)), mode="constant", constant_values=0.0)
        y = np.pad(y, ((0, 0), (bd, ad), (bh, ah), (bw, aw)), mode="constant", constant_values=0.0)
        return x, y

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=2).copy()
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=3).copy()
            y = np.flip(y, axis=3).copy()

        if np.random.rand() < 0.3:
            k = np.random.randint(1, 4)
            x = np.rot90(x, k, axes=(2, 3)).copy()
            y = np.rot90(y, k, axes=(2, 3)).copy()

        if np.random.rand() < 0.3:
            noise = np.random.normal(0.0, 0.05, size=x[0].shape).astype(np.float32)
            x[0] = x[0] + noise

        if np.random.rand() < 0.3:
            scale = float(np.clip(1.0 + np.random.normal(0.0, 0.05), 0.8, 1.2))
            shift = float(np.random.normal(0.0, 0.05))
            x[0] = x[0] * scale + shift

        if binary_dilation is not None and np.random.rand() < 0.3:
            mask = y[0] > 0.5
            if mask.any():
                struct = np.ones((3, 3, 3), dtype=bool)
                y[0] = binary_dilation(mask, structure=struct).astype(np.float32)

        return x, y

    def _meta_for_patient(self, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
        info = self.add_lookup.get(patient_id, {})

        cat_idx: List[int] = []
        for k in self.cat_order:
            vocab = self.vocabs[k]
            if k in (KEY_CORTEX, KEY_IDH, KEY_HISTO):
                cat_idx.append(map_to_index_str(info.get(k, UNK), vocab))
            else:
                raise RuntimeError(f"Unexpected categorical key: {k}")

        cat_arr = np.array(cat_idx, dtype=np.int64) if len(cat_idx) > 0 else np.zeros((0,), dtype=np.int64)
        return cat_arr

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patient_id = row[constants.KEY_PATIENT_ID]

        baseline = self._load_nii(row[constants.KEY_BASELINE_IMG])
        baseline = self._zscore_normalize(baseline)

        fu_mask = self._load_nii(row[constants.KEY_FOLLOWUPS_MASK])
        fu_mask = (fu_mask > 0.5).astype(np.float32)

        if row[constants.KEY_BASELINE_MASK] is not None:
            residual = self._load_nii(row[constants.KEY_BASELINE_MASK])
            residual = (residual > 0.5).astype(np.float32)
        else:
            residual = np.zeros_like(baseline, dtype=np.float32)

        if baseline.shape != fu_mask.shape:
            raise ValueError(f"Shape mismatch: {row[constants.KEY_BASELINE_IMG]} {baseline.shape} vs {row[constants.KEY_FOLLOWUPS_MASK]} {fu_mask.shape}")
        if baseline.shape != residual.shape:
            raise ValueError(f"Shape mismatch: {row[constants.KEY_BASELINE_IMG]} {baseline.shape} vs {row[constants.KEY_BASELINE_MASK]} {residual.shape}")

        day_value = row[constants.KEY_DAY]
        t = float(day_value) if pd.notna(day_value) else 0.0
        t = max(0.0, t)
        t_norm = np.log1p(t) / np.log1p(self.t_max_days)
        t_norm = float(np.clip(t_norm, 0.0, 1.0))
        t_vol = np.full_like(baseline, t_norm, dtype=np.float32)

        x = np.stack([baseline, residual, t_vol], axis=0)
        y = fu_mask[None, ...]

        x, y = self._pad_to_multiple(x, y)
        if self.augment:
            x, y = self._augment(x, y)

        x_t = torch.from_numpy(x)                   
        y_t = torch.from_numpy(y)   

        if not self.use_film:
            return x_t, y_t
        
        meta_cat = self._meta_for_patient(patient_id)
        mc_t = torch.from_numpy(meta_cat)       

        return x_t, mc_t, y_t
