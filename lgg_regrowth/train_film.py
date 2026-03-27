import os
from typing import Dict, List

import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader

import constants
from core.config import parse_args_film
from core.data import (
    KEY_CORTEX,
    KEY_HISTO,
    KEY_IDH,
    UNK,
    TumourRegrowthDataset,
    build_vocab,
    compute_t_max_days,
    enabled_feature_keys,
    get_df,
    get_train_val_split,
    load_additional_info,
    make_addinfo_lookup,
    print_distinct_values,
)
from core.metrics import (
    residual_inclusion_loss_from_prob,
    volume_consistency_loss_from_logits,
)
from core.models import ResizeConvUNet3D_FiLM
from core.train_loop import run_training, set_all_seeds


def train(args):
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    enabled = enabled_feature_keys()
    print("[FiLM] Enabled features:", {k: v for k, v in enabled.items() if v})

    df = get_df(args.data_root)
    if len(df) == 0:
        raise RuntimeError(f"No samples found under {args.data_root} with expected pattern.")
    print("Total samples:", len(df))
    print("Unique patients:", df[constants.KEY_PATIENT_ID].nunique())

    add_df = load_additional_info(args.data_root)
    print_distinct_values(add_df)
    add_lookup = make_addinfo_lookup(add_df)

    vocabs: Dict[str, Dict[str, int]] = {}
    cat_keys_in_order: List[str] = []

    if enabled.get(KEY_CORTEX, False):
        vocabs[KEY_CORTEX] = build_vocab(add_df[KEY_CORTEX].tolist())
        cat_keys_in_order.append(KEY_CORTEX)

    if enabled.get(KEY_IDH, False):
        vocabs[KEY_IDH] = build_vocab(add_df[KEY_IDH].tolist())
        cat_keys_in_order.append(KEY_IDH)

    if enabled.get(KEY_HISTO, False):
        vocabs[KEY_HISTO] = build_vocab(add_df[KEY_HISTO].tolist())
        cat_keys_in_order.append(KEY_HISTO)

    print("[FiLM] Categorical order:", cat_keys_in_order)
    print("[FiLM] Vocab sizes:", {k: len(v) for k, v in vocabs.items()})

    for k, v in vocabs.items():
        if UNK not in v:
            raise RuntimeError(f"UNK token missing from vocab for {k} (should not happen).")

    df_train, df_val = get_train_val_split(
        df,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print("Train samples:", len(df_train), "Val samples:", len(df_val))

    t_max_days = compute_t_max_days(df_train)

    train_ds = TumourRegrowthDataset(
        df_train,
        augment=args.augment,
        t_max_days=t_max_days,
        add_lookup=add_lookup,
        vocabs=vocabs,
        enabled=enabled,
    )
    val_ds = TumourRegrowthDataset(
        df_val,
        augment=False,
        t_max_days=t_max_days,
        add_lookup=add_lookup,
        vocabs=vocabs,
        enabled=enabled,
    )

    pin_memory = torch.cuda.is_available()
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty after split.")
    x0, meta_cat0, y0 = train_ds[0]
    print("Example sample shapes:", x0.shape, meta_cat0.shape, y0.shape)
    print("[FiLM] Dataset cat_order:", train_ds.cat_order)

    cat_vocab_sizes = [len(vocabs[k]) for k in train_ds.cat_order]

    model = ResizeConvUNet3D_FiLM(
        in_channels=3,
        out_channels=1,
        channels=tuple(constants.CHANNELS),
        cat_vocab_sizes=cat_vocab_sizes,
        meta_emb_dim=args.meta_emb_dim,
        meta_dim=args.meta_dim,
        meta_dropout=args.meta_dropout,
    ).to(device)

    dice_loss_fn = DiceLoss(
        include_background=True,
        to_onehot_y=False,
        sigmoid=True,
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            threshold=1e-4,
            verbose=True,
        )

    use_amp = torch.cuda.is_available() and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, constants.MODEL_FILE_NAME)

    def build_ckpt_fn():
        return {
            constants.CKPT_STATE_DICT: model.state_dict(),
            constants.CKPT_MAX_DAYS: float(t_max_days),
            constants.CKPT_PAD_MULTIPLE: int(constants.PAD_MULTIPLE),
            constants.CKPT_CHANNELS: list(constants.CHANNELS),
            constants.CKPT_SEED: int(args.seed),
            constants.CKPT_META_EMB_DIM: int(args.meta_emb_dim),
            constants.CKPT_META_DIM: int(args.meta_dim),
            constants.CKPT_META_DROPOUT: float(args.meta_dropout),
            constants.CKPT_FILM: enabled,
            constants.CKPT_CAT_ORDER: list(train_ds.cat_order),
            constants.CKPT_VOCABS: vocabs,
        }

    return run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        dice_loss_fn=dice_loss_fn,
        bce_loss_fn=bce_loss_fn,
        volume_consistency_loss_from_logits=volume_consistency_loss_from_logits,
        residual_inclusion_loss_from_prob=residual_inclusion_loss_from_prob,
        device=device,
        epochs=args.epochs,
        lambda_vol=args.lambda_vol,
        lambda_res=args.lambda_res,
        grad_clip=args.grad_clip,
        use_amp=use_amp,
        ckpt_path=ckpt_path,
        build_ckpt_fn=build_ckpt_fn,
        early_stop_patience=args.early_stop_patience,
        use_film=True,
    )


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(0))

    args = parse_args_film()
    train(args)