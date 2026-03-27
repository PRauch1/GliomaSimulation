from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import constants

# ==========================
# FiLM components
# ==========================

class FiLM(nn.Module):
    def __init__(self, meta_dim: int, num_features: int):
        super().__init__()
        self.linear = nn.Linear(meta_dim, 2 * num_features)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, meta_vec: torch.Tensor) -> torch.Tensor:
        params = self.linear(meta_vec)  # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return x * (1.0 + gamma) + beta

class MetaEncoder(nn.Module):
    def __init__(
        self,
        cat_vocab_sizes: List[int],
        emb_dim: int = 16,
        meta_dim: int = 64,
        dropout: float = 0.0,
        n_cont: int = 0,
    ):
        super().__init__()
        self.n_cat = len(cat_vocab_sizes)
        self.n_cont = int(n_cont)

        self.embs = nn.ModuleList([nn.Embedding(vs, emb_dim) for vs in cat_vocab_sizes])

        in_dim = self.n_cat * emb_dim + self.n_cont
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, meta_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(meta_dim, meta_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, meta_cat_idx: torch.Tensor) -> torch.Tensor:
        zs = []
        if self.n_cat > 0:
            for i, emb in enumerate(self.embs):
                zs.append(emb(meta_cat_idx[:, i].long()))
        z = torch.cat(zs, dim=1) if len(zs) > 1 else zs[0]
        return self.mlp(z)

# ==========================
# Basic building blocks
# ==========================
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, meta_dim: Optional[int] = None):
        super().__init__()
        self.use_film = meta_dim is not None

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        if self.use_film:
            self.film1 = FiLM(meta_dim, out_ch)
            self.film2 = FiLM(meta_dim, out_ch)

    def forward(self, x: torch.Tensor, meta_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        if self.use_film:
            if meta_vec is None:
                raise ValueError("meta_vec is required when ConvBlock3D uses FiLM")
            x = self.film1(x, meta_vec)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if self.use_film:
            x = self.film2(x, meta_vec)
        x = self.act2(x)
        return x

class UpBlock3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, meta_dim: Optional[int] = None):
        super().__init__()
        self.use_film = meta_dim is not None

        self.reduce_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.reduce_norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.reduce_act = nn.LeakyReLU(0.01, inplace=True)

        if self.use_film:
            self.reduce_film = FiLM(meta_dim, out_ch)

        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch, meta_dim=meta_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        meta_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.reduce_conv(x)
        x = self.reduce_norm(x)
        if self.use_film:
            if meta_vec is None:
                raise ValueError("meta_vec is required when UpBlock3D uses FiLM")
            x = self.reduce_film(x, meta_vec)
        x = self.reduce_act(x)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x, meta_vec)

class ResizeConvUNet3D(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, channels=(16, 32, 64, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.enc1 = ConvBlock3D(in_channels, c1)
        self.down1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(c1, c2)
        self.down2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(c2, c3)
        self.down3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(c3, c4)

        self.up3 = UpBlock3D(in_ch=c4, skip_ch=c3, out_ch=c3)
        self.up2 = UpBlock3D(in_ch=c3, skip_ch=c2, out_ch=c2)
        self.up1 = UpBlock3D(in_ch=c2, skip_ch=c1, out_ch=c1)

        self.out = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x); x = self.down1(s1)
        s2 = self.enc2(x); x = self.down2(s2)
        s3 = self.enc3(x); x = self.down3(s3)

        x = self.bottleneck(x)

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out(x)
    
class ResizeConvUNet3D_FiLM(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        channels=(16, 32, 64, 128),
        cat_vocab_sizes: Optional[List[int]] = None,
        meta_emb_dim: int = 16,
        meta_dim: int = 64,
        meta_dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels

        cat_vocab_sizes = cat_vocab_sizes or []
        self.meta_enc = MetaEncoder(
            cat_vocab_sizes=cat_vocab_sizes,
            emb_dim=meta_emb_dim,
            meta_dim=meta_dim,
            dropout=meta_dropout,
        )

        self.enc1 = ConvBlock3D(in_channels, c1, meta_dim)
        self.down1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(c1, c2, meta_dim)
        self.down2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(c2, c3, meta_dim)
        self.down3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(c3, c4, meta_dim)

        self.up3 = UpBlock3D(in_ch=c4, skip_ch=c3, out_ch=c3, meta_dim=meta_dim)
        self.up2 = UpBlock3D(in_ch=c3, skip_ch=c2, out_ch=c2, meta_dim=meta_dim)
        self.up1 = UpBlock3D(in_ch=c2, skip_ch=c1, out_ch=c1, meta_dim=meta_dim)

        self.out = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, meta_cat_idx: torch.Tensor) -> torch.Tensor:
        meta_vec = self.meta_enc(meta_cat_idx)  # (B, meta_dim)

        s1 = self.enc1(x, meta_vec); x = self.down1(s1)
        s2 = self.enc2(x, meta_vec); x = self.down2(s2)
        s3 = self.enc3(x, meta_vec); x = self.down3(s3)

        x = self.bottleneck(x, meta_vec)

        x = self.up3(x, s3, meta_vec)
        x = self.up2(x, s2, meta_vec)
        x = self.up1(x, s1, meta_vec)
        return self.out(x)
