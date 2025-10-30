import math
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.utils import save_image
from transformers import AutoConfig

from stage1.decoders import GeneralDecoder


class FeatureNormalizer:
    """Applies latent-space normalization using statistics estimated from ImageNet."""

    def __init__(self, stats_path: Optional[str], eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.var: Optional[torch.Tensor] = None
        if stats_path is not None and Path(stats_path).exists():
            stats = torch.load(stats_path, map_location="cpu")
            self.mean = stats.get("mean", None)
            self.var = stats.get("var", None)

    def to(self, device: torch.device) -> "FeatureNormalizer":
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.var is not None:
            self.var = self.var.to(device)
        return self

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        if self.mean is None and self.var is None:
            return features
        reshaped = self._to_hw(features)
        if self.mean is not None:
            reshaped = reshaped - self.mean
        if self.var is not None:
            reshaped = reshaped / torch.sqrt(self.var + self.eps)
        return self._to_tokens(reshaped)

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        if self.mean is None and self.var is None:
            return features
        reshaped = self._to_hw(features)
        if self.var is not None:
            reshaped = reshaped * torch.sqrt(self.var + self.eps)
        if self.mean is not None:
            reshaped = reshaped + self.mean
        return self._to_tokens(reshaped)

    @staticmethod
    def _to_hw(features: torch.Tensor) -> torch.Tensor:
        b, n, c = features.shape
        h = w = int(math.sqrt(n))
        return features.transpose(1, 2).reshape(b, c, h, w)

    @staticmethod
    def _to_tokens(features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        n = h * w
        return features.reshape(b, c, n).transpose(1, 2)


def build_rae_decoder(
    config_path: str,
    checkpoint_path: str,
    *,
    hidden_size: int,
    num_patches: int,
    device: torch.device,
) -> nn.Module:
    decoder_config = AutoConfig.from_pretrained(config_path)
    decoder_config.hidden_size = hidden_size
    decoder = GeneralDecoder(decoder_config, num_patches=num_patches)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Decoder] Missing keys: {missing}")
    if unexpected:
        print(f"[Decoder] Unexpected keys: {unexpected}")
    decoder.to(device)
    decoder.eval()
    return decoder


def save_reconstruction_grid(
    outputs: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    filename: Path,
    nrow: int = 4,
) -> None:
    denorm = outputs * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    denorm = torch.clamp(denorm, 0.0, 1.0)
    filename.parent.mkdir(parents=True, exist_ok=True)
    save_image(denorm, filename, nrow=nrow)
