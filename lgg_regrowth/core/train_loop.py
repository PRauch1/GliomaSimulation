import os
from typing import Any, Dict, Optional

import numpy as np
import torch

def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def forward_model(model, batch, device: torch.device, use_film: bool):
    if use_film:
        x, meta_cat, y = batch
        x = x.to(device, non_blocking=True)
        meta_cat = meta_cat.to(device, non_blocking=True).long()
        y = y.to(device, non_blocking=True)
        logits = model(x, meta_cat)
        return x, y, logits
    else:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        return x, y, logits


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    dice_loss_fn,
    bce_loss_fn,
    volume_consistency_loss_from_logits,
    residual_inclusion_loss_from_prob,
    device: torch.device,
    use_amp: bool,
    lambda_vol: float,
    lambda_res: float,
    grad_clip: float = 0.0,
    use_film: bool = False,
) -> Dict[str, float]:
    model.train()
    running_total = 0.0

    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            x, y, logits = forward_model(model, batch, device=device, use_film=use_film)

            seg_loss = dice_loss_fn(logits, y) + bce_loss_fn(logits, y)
            vol_loss = volume_consistency_loss_from_logits(logits, y)

            residual = x[:, 1:2].clamp(0, 1)
            p = torch.sigmoid(logits)
            res_loss = residual_inclusion_loss_from_prob(p, residual)

            loss = seg_loss + lambda_vol * vol_loss + lambda_res * res_loss

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_total += float(loss.item())

    avg_train_total = running_total / max(1, len(train_loader))
    return {
        "train_total": avg_train_total,
    }

@torch.no_grad()
def validate_one_epoch(
    model,
    val_loader,
    dice_loss_fn,
    bce_loss_fn,
    volume_consistency_loss_from_logits,
    residual_inclusion_loss_from_prob,
    device: torch.device,
    use_amp: bool,
    lambda_vol: float,
    lambda_res: float,
    use_film: bool = False,
) -> Dict[str, float]:
    model.eval()

    val_total_running = 0.0
    val_ckpt_running = 0.0
    val_res_running = 0.0

    for batch in val_loader:
        with torch.cuda.amp.autocast(enabled=use_amp):
            x, y, logits = forward_model(model, batch, device=device, use_film=use_film)

            seg_loss_val = dice_loss_fn(logits, y) + bce_loss_fn(logits, y)
            vol_loss_val = volume_consistency_loss_from_logits(logits, y)

            residual_val = x[:, 1:2].clamp(0, 1)
            p_val = torch.sigmoid(logits)
            res_loss_val = residual_inclusion_loss_from_prob(p_val, residual_val)

            loss_val_total = seg_loss_val + lambda_vol * vol_loss_val + lambda_res * res_loss_val
            loss_val_ckpt = seg_loss_val + lambda_vol * vol_loss_val

        val_total_running += float(loss_val_total.item())
        val_ckpt_running += float(loss_val_ckpt.item())
        val_res_running += float(res_loss_val.item())

    avg_val_total = val_total_running / max(1, len(val_loader))
    avg_val_ckpt = val_ckpt_running / max(1, len(val_loader))
    avg_val_res = val_res_running / max(1, len(val_loader))

    return {
        "val_total": avg_val_total,
        "val_ckpt": avg_val_ckpt,
        "val_res": avg_val_res,
    }

def save_checkpoint(ckpt: Dict[str, Any], ckpt_path: str) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(ckpt, ckpt_path)

def run_training(
    *,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    dice_loss_fn,
    bce_loss_fn,
    volume_consistency_loss_from_logits,
    residual_inclusion_loss_from_prob,
    device: torch.device,
    epochs: int,
    lambda_vol: float,
    lambda_res: float,
    grad_clip: float,
    use_amp: bool,
    ckpt_path: str,
    build_ckpt_fn,
    early_stop_patience: int = 7,
    min_delta: float = 1e-4,
    use_film: bool = False,
) -> Dict[str, float]:
    best_val_ckpt = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            dice_loss_fn=dice_loss_fn,
            bce_loss_fn=bce_loss_fn,
            volume_consistency_loss_from_logits=volume_consistency_loss_from_logits,
            residual_inclusion_loss_from_prob=residual_inclusion_loss_from_prob,
            device=device,
            use_amp=use_amp,
            lambda_vol=lambda_vol,
            lambda_res=lambda_res,
            grad_clip=grad_clip,
            use_film=use_film,
        )

        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            dice_loss_fn=dice_loss_fn,
            bce_loss_fn=bce_loss_fn,
            volume_consistency_loss_from_logits=volume_consistency_loss_from_logits,
            residual_inclusion_loss_from_prob=residual_inclusion_loss_from_prob,
            device=device,
            use_amp=use_amp,
            lambda_vol=lambda_vol,
            lambda_res=lambda_res,
            use_film=use_film,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | lr={lr_now:.2e} | "
            f"train_total={train_metrics['train_total']:.4f} | "
            f"val_ckpt(seg+vol)={val_metrics['val_ckpt']:.4f} | "
            f"val_total={val_metrics['val_total']:.4f} | "
            f"val_res={val_metrics['val_res']:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_metrics["val_ckpt"])

        if val_metrics["val_ckpt"] < best_val_ckpt - min_delta:
            best_val_ckpt = val_metrics["val_ckpt"]
            no_improve = 0

            ckpt = build_ckpt_fn()
            save_checkpoint(ckpt, ckpt_path)
            print("  New best saved:", ckpt_path, f"(val_ckpt={best_val_ckpt:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{early_stop_patience}) best_val_ckpt={best_val_ckpt:.4f}")

            if no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}. Best val_ckpt={best_val_ckpt:.4f}")
                break

        row = {
            "epoch": epoch,
            "lr": lr_now,
            **train_metrics,
            **val_metrics,
        }
        history.append(row)

    return {
        "best_val_ckpt": best_val_ckpt,
        "history": history,
    }