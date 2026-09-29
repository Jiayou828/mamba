"""Spatial and frequency-domain losses from the DehazeMamba paper."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def dehaze_loss(prediction: Tensor, target: Tensor, frequency_weight: float = 0.1):
    spatial = F.l1_loss(prediction, target)
    # Orthonormal FFT keeps the frequency term on a scale comparable to image L1.
    pred_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    frequency = F.l1_loss(torch.view_as_real(pred_fft), torch.view_as_real(target_fft))
    total = spatial + frequency_weight * frequency
    return total, spatial.detach(), frequency.detach()
