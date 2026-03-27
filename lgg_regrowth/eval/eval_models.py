from typing import Any, Dict, Optional

import torch

import constants
from eval.eval_metadata import load_additional_info, make_addinfo_lookup, print_distinct_for_check
from core.models import ResizeConvUNet3D, ResizeConvUNet3D_FiLM


def build_baseline_model(device: Optional[torch.device], model_path: str) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResizeConvUNet3D(
        in_channels=3,
        out_channels=1,
        channels=tuple(constants.CHANNELS),
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and constants.CKPT_STATE_DICT in ckpt:
        state = ckpt[constants.CKPT_STATE_DICT]
        t_max_days = float(ckpt.get(constants.CKPT_MAX_DAYS, 0.0)) or None
    else:
        state = ckpt
        t_max_days = None

    model.load_state_dict(state, strict=True)
    model.eval()

    return {
        constants.CKPT_MODEL: model,
        constants.CKPT_DEVICE: device,
        constants.CKPT_MAX_DAYS: t_max_days,
        constants.CKPT_META_BUNDLE: None,
        constants.CKPT_ADD_LOOKUP: None,
    }


def build_film_model(
    device: Optional[torch.device],
    model_path: str,
    data_root: str,
    print_distinct_metadata: bool = False,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)

    if not (isinstance(ckpt, dict) and constants.CKPT_STATE_DICT in ckpt):
        raise ValueError(
            "Expected a checkpoint dict with CKPT_STATE_DICT. "
            "Point MODEL_PATH at best_unet_film.pth (or equivalent)."
        )

    state = ckpt[constants.CKPT_STATE_DICT]
    t_max_days = float(ckpt.get(constants.CKPT_MAX_DAYS, 0.0)) or None

    enabled = ckpt.get(constants.CKPT_FILM, None)
    cat_order = ckpt.get(constants.CKPT_CAT_ORDER, None)
    vocabs = ckpt.get(constants.CKPT_VOCABS, None)

    if enabled is None or cat_order is None or vocabs is None:
        raise ValueError(
            "Checkpoint missing film metadata fields. Expected: "
            "film_enabled, cat_order, cont_order, vocabs, age_mean, age_std."
        )

    meta_emb_dim = int(ckpt.get(constants.CKPT_META_EMB_DIM, 16))
    meta_dim = int(ckpt.get(constants.CKPT_META_DIM, 64))
    meta_dropout = float(ckpt.get(constants.CKPT_META_DROPOUT, 0.0))

    cat_vocab_sizes = [len(vocabs[k]) for k in cat_order]

    model = ResizeConvUNet3D_FiLM(
        in_channels=3,
        out_channels=1,
        channels=tuple(constants.CHANNELS),
        cat_vocab_sizes=cat_vocab_sizes,
        meta_emb_dim=meta_emb_dim,
        meta_dim=meta_dim,
        meta_dropout=meta_dropout,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    required_cols = sorted(set(cat_order))
    add_df = load_additional_info(data_root, required_cols=required_cols)
    if print_distinct_metadata:
        print_distinct_for_check(add_df, cat_order)
    add_lookup = make_addinfo_lookup(add_df)

    meta_bundle = {
        constants.CKPT_ENABLED: enabled,
        constants.CKPT_CAT_ORDER: list(cat_order),
        constants.CKPT_VOCABS: vocabs,
        constants.CKPT_META_CONFIG: {
            constants.CKPT_META_EMB_DIM: meta_emb_dim,
            constants.CKPT_META_DIM: meta_dim,
            constants.CKPT_META_DROPOUT: meta_dropout,
        },
    }

    return {
        constants.CKPT_MODEL: model,
        constants.CKPT_DEVICE: device,
        constants.CKPT_MAX_DAYS: t_max_days,
        constants.CKPT_META_BUNDLE: meta_bundle,
        constants.CKPT_ADD_LOOKUP: add_lookup,
    }