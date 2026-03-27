import os

import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader

import constants
from core.config import parse_args_baseline
from core.data import (
    TumourRegrowthDataset,
    compute_t_max_days,
    get_df,
    get_train_val_split,
)
from core.metrics import (
    residual_inclusion_loss_from_prob,
    volume_consistency_loss_from_logits,
)
from core.models import ResizeConvUNet3D
from core.train_loop import run_training, set_all_seeds


def train(args):
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = get_df(args.data_root)
    if len(df) == 0:
        raise RuntimeError(f"No samples found under {args.data_root} with expected pattern.")
    print("Total samples:", len(df))
    print("Unique patients:", df[constants.KEY_PATIENT_ID].nunique())

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
    )
    val_ds = TumourRegrowthDataset(
        df_val,
        augment=False,
        t_max_days=t_max_days,
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
    x0, y0 = train_ds[0]
    print("Example sample shapes:", x0.shape, y0.shape)

    model = ResizeConvUNet3D(
        in_channels=3,
        out_channels=1,
        channels=tuple(constants.CHANNELS),
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
        use_film=False,
    )


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(0))

    args = parse_args_baseline()
    train(args)