"""PyTorch implementation of the DehazeMamba DM-T network.

The paper specifies the five stage depths and the HPDM/PFM operations, but does
not publish the channel widths or complete VSS implementation. Those choices
are exposed in the constructor so the reproduction assumptions are explicit.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mamba_ssm.modules.mamba_simple import Mamba


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class VSS2D(nn.Module):
    """Axial bidirectional selective scan with shared Mamba weights.

    Four scans (left/right and top/bottom) give each pixel context along both
    spatial axes without forming a 512x512 sequence in one CUDA scan.
    """

    def __init__(self, channels: int, state_dim: int = 8, expand: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.scan = Mamba(
            d_model=channels,
            d_state=state_dim,
            d_conv=4,
            expand=expand,
            use_fast_path=False,
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def _run_scan(self, sequence: Tensor) -> Tensor:
        if self.training and torch.is_grad_enabled() and sequence.requires_grad:
            return checkpoint(self.scan, sequence, use_reentrant=True)
        return self.scan(sequence)

    def _bidirectional(self, sequence: Tensor) -> Tensor:
        forward = self._run_scan(sequence)
        reverse = torch.flip(self._run_scan(torch.flip(sequence, dims=(1,))), dims=(1,))
        return 0.5 * (forward + reverse)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        grid = x.permute(0, 2, 3, 1).contiguous()
        rows = grid.reshape(batch * height, width, channels)
        cols = grid.transpose(1, 2).contiguous().reshape(batch * width, height, channels)

        row_features = self._bidirectional(rows).reshape(batch, height, width, channels)
        col_features = self._bidirectional(cols).reshape(batch, width, height, channels)
        col_features = col_features.transpose(1, 2)
        merged = 0.5 * (row_features + col_features)
        return self.proj(merged.permute(0, 3, 1, 2).contiguous())


class DMBlock(nn.Module):
    """VSS + local MLP block with learnable residual scales (paper Fig. 3)."""

    def __init__(self, channels: int, state_dim: int = 8, expand: int = 1) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.vss = VSS2D(channels, state_dim=state_dim, expand=expand)
        self.scale1 = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))
        self.norm2 = LayerNorm2d(channels)
        hidden = channels * 2
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.scale2 = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.scale1 * self.vss(self.norm1(x))
        return x + self.scale2 * self.mlp(self.norm2(x))


class HPDM(nn.Module):
    """Haze Perception and Decoupling Module (paper Algorithm 1)."""

    def __init__(self, channels: int, state_dim: int = 8) -> None:
        super().__init__()
        self.extract_rgb = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
        )
        self.extract_sar = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
        )
        self.css_rgb = VSS2D(channels, state_dim=state_dim)
        self.css_sar = VSS2D(channels, state_dim=state_dim)
        self.gate_rgb = nn.Conv2d(channels, channels, 1)
        self.gate_sar = nn.Conv2d(channels, channels, 1)
        self.weight = nn.Conv2d(channels, channels, 1)

    def forward(self, rgb: Tensor, sar: Tensor) -> tuple[Tensor, Tensor]:
        rgb_feature = self.extract_rgb(rgb)
        sar_feature = self.extract_sar(sar)
        difference = (self.css_rgb(rgb_feature) - self.css_sar(sar_feature)).abs()
        joint_gate = F.silu(self.gate_rgb(rgb) + self.gate_sar(sar))
        haze_weight = torch.sigmoid(self.weight(joint_gate * difference))
        return (1.0 - haze_weight) * rgb, haze_weight * sar


class PFM(nn.Module):
    """Progressive Fusion Module (paper Algorithm 2)."""

    def __init__(self, channels: int, state_dim: int = 8) -> None:
        super().__init__()
        self.coarse = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.refine = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.vss = VSS2D(channels, state_dim=state_dim)
        self.output = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, rgb: Tensor, sar: Tensor) -> Tensor:
        coarse = self.coarse(torch.cat((rgb, sar), dim=1))
        weight = torch.sigmoid(coarse)
        rgb_adapted = weight * rgb
        sar_adapted = (1.0 - weight) * sar
        refined = self.refine(torch.cat((rgb_adapted, sar_adapted, coarse), dim=1))
        return self.output(self.vss(refined))


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class SKFusion(nn.Module):
    """Spatially adaptive skip fusion for encoder/decoder features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weights = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        b, c, h, w = x.shape
        weights = self.weights(torch.cat((x, skip), dim=1)).reshape(b, 2, c, h, w)
        weights = torch.softmax(weights, dim=1)
        return weights[:, 0] * x + weights[:, 1] * skip


class DehazeMamba(nn.Module):
    """Dual-branch DM-T encoder/fusion/decoder network.

    ``depths`` are the DM-T stage counts reported by the paper. ``base_channels``
    and ``state_dim`` are configurable because the article does not report them.
    """

    def __init__(
        self,
        base_channels: int = 24,
        state_dim: int = 8,
        depths: Sequence[int] = (2, 2, 2, 1, 1),
    ) -> None:
        super().__init__()
        if len(depths) != 5:
            raise ValueError("DehazeMamba uses five encoder stages")
        self.base_channels = base_channels
        self.state_dim = state_dim
        self.depths = tuple(int(depth) for depth in depths)
        widths = [base_channels * (2**stage) for stage in range(5)]
        self.widths = tuple(widths)

        self.rgb_stem = nn.Conv2d(3, widths[0], kernel_size=3, padding=1)
        self.sar_stem = nn.Conv2d(1, widths[0], kernel_size=3, padding=1)
        self.rgb_stages = nn.ModuleList()
        self.sar_stages = nn.ModuleList()
        for channels, depth in zip(widths, self.depths):
            self.rgb_stages.append(
                nn.Sequential(*(DMBlock(channels, state_dim) for _ in range(depth)))
            )
            self.sar_stages.append(
                nn.Sequential(*(DMBlock(channels, state_dim) for _ in range(depth)))
            )
        self.rgb_down = nn.ModuleList(
            Downsample(widths[i], widths[i + 1]) for i in range(4)
        )
        self.sar_down = nn.ModuleList(
            Downsample(widths[i], widths[i + 1]) for i in range(4)
        )

        # The paper applies selective RGB-SAR fusion at the two deepest scales.
        self.hpdm = nn.ModuleDict({str(i): HPDM(widths[i], state_dim) for i in (3, 4)})
        self.pfm = nn.ModuleDict({str(i): PFM(widths[i], state_dim) for i in (3, 4)})
        self.fusion_refine = nn.ModuleDict(
            {str(i): DMBlock(widths[i], state_dim) for i in (3, 4)}
        )

        self.upsample = nn.ModuleList(
            Upsample(widths[i + 1], widths[i]) for i in range(4)
        )
        self.skip_fusion = nn.ModuleList(SKFusion(widths[i]) for i in range(4))
        self.decoder = nn.ModuleList(
            DMBlock(widths[i], state_dim) for i in range(4)
        )
        self.head = nn.Conv2d(widths[0], 3, kernel_size=3, padding=1)

    def forward(self, hazy_rgb: Tensor, sar: Tensor) -> Tensor:
        if hazy_rgb.ndim != 4 or hazy_rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB input [B,3,H,W], got {tuple(hazy_rgb.shape)}")
        if sar.ndim != 4 or sar.shape[1] != 1:
            raise ValueError(f"Expected SAR input [B,1,H,W], got {tuple(sar.shape)}")
        if hazy_rgb.shape[0] != sar.shape[0] or hazy_rgb.shape[-2:] != sar.shape[-2:]:
            raise ValueError("RGB and SAR inputs must have matching batch and spatial sizes")

        height, width = hazy_rgb.shape[-2:]
        pad_h = (-height) % 16
        pad_w = (-width) % 16
        if pad_h or pad_w:
            pad = (0, pad_w, 0, pad_h)
            hazy_rgb = F.pad(hazy_rgb, pad, mode="replicate")
            sar = F.pad(sar, pad, mode="replicate")

        rgb = self.rgb_stem(hazy_rgb)
        sar_feature = self.sar_stem(sar)
        skips: list[Tensor] = []
        for stage in range(5):
            rgb = self.rgb_stages[stage](rgb)
            sar_feature = self.sar_stages[stage](sar_feature)
            if stage in (3, 4):
                rgb_decoupled, sar_decoupled = self.hpdm[str(stage)](rgb, sar_feature)
                rgb = self.fusion_refine[str(stage)](
                    rgb + self.pfm[str(stage)](rgb_decoupled, sar_decoupled)
                )
            skips.append(rgb)
            if stage < 4:
                rgb = self.rgb_down[stage](rgb)
                sar_feature = self.sar_down[stage](sar_feature)

        decoded = skips[-1]
        for decoder_index, encoder_index in enumerate(reversed(range(4))):
            decoded = self.upsample[encoder_index](decoded)
            decoded = self.skip_fusion[encoder_index](decoded, skips[encoder_index])
            decoded = self.decoder[encoder_index](decoded)

        output = hazy_rgb + self.head(decoded)
        return output[..., :height, :width]
