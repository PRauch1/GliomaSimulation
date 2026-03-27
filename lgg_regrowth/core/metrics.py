import torch

def volume_consistency_loss_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)

    v_pred = p.sum(dim=(1, 2, 3, 4))
    v_true = y.sum(dim=(1, 2, 3, 4))

    vox = y[0, 0].numel()
    vox_t = torch.as_tensor(vox, device=y.device, dtype=p.dtype)

    v_pred_n = v_pred / (vox_t + 1e-6)
    v_true_n = v_true / (vox_t + 1e-6)

    has_tumour = v_true > 0
    loss_pos = (v_pred_n[has_tumour] - v_true_n[has_tumour]).abs().mean() if has_tumour.any() else torch.zeros((), device=y.device, dtype=p.dtype)
    loss_zero = v_pred_n[~has_tumour].mean() if (~has_tumour).any() else torch.zeros((), device=y.device, dtype=p.dtype)

    return 0.5 * (loss_pos + loss_zero)


def residual_inclusion_loss_from_prob(p: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    residual = residual.clamp(0, 1)
    if residual.sum() == 0:
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return ((1.0 - p) * residual).sum() / (residual.sum() + 1e-6)