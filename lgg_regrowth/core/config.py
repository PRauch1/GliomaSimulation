import argparse

import constants

def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--data_root", type=str, default="dataset",
                    help="Root directory containing Patient_*/op_*/preprocessed/meta.json")
    ap.add_argument("--out_dir", type=str, default=constants.DIR_MODEL,
                    help="Directory to save model checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--augment", action="store_true",
                    help="Enable data augmentation in training")
    ap.add_argument("--lambda_vol", type=float, default=0.0,
                    help="Weight for safe volume consistency loss")
    ap.add_argument("--lambda_res", type=float, default=0.05,
                    help="Weight for soft residual inclusion loss (train only for optimization; logged in val)")
    ap.add_argument("--early_stop_patience", type=int, default=7)
    ap.add_argument("--grad_clip", type=float, default=0.0,
                    help="Clip grad norm if > 0")
    ap.add_argument("--use_scheduler", action="store_true",
                    help="Enable ReduceLROnPlateau scheduler")
    ap.add_argument("--no_amp", action="store_true",
                    help="Disable AMP even if CUDA is available")

    return ap

def add_film_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # metadata encoder sizes
    ap.add_argument("--meta_emb_dim", type=int, default=16,
                    help="Embedding dim per categorical field")
    ap.add_argument("--meta_dim", type=int, default=64,
                    help="Meta vector dim fed into FiLM")
    ap.add_argument("--meta_dropout", type=float, default=0.0,
                    help="Dropout in metadata encoder MLP")

    return ap.parse_args()


def parse_args_baseline():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    return ap.parse_args()


def parse_args_film():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    add_film_args(ap)
    return ap.parse_args()