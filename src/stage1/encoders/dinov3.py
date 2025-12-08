import os
import torch
from torch import nn
from transformers import DINOv3ViTModel
from . import register_encoder


def _get_hf_token() -> str | None:
    """Return a Hugging Face auth token from known environment variables."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


@register_encoder()
class Dinov3withNorm(nn.Module):
    def __init__(
        self,
        dinov3_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        token = _get_hf_token()
        try:
            self.encoder = DINOv3ViTModel.from_pretrained(dinov3_path, local_files_only=True, token=token)
        except (OSError, ValueError, AttributeError):
            self.encoder = DINOv3ViTModel.from_pretrained(dinov3_path, local_files_only=False, token=token)
        self.encoder.requires_grad_(False)
        if normalize and hasattr(self.encoder, "norm") and isinstance(self.encoder.norm, nn.LayerNorm):
            self.encoder.norm.elementwise_affine = False
            self.encoder.norm.weight = None
            self.encoder.norm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self._unused_tokens = 1 + int(getattr(self.encoder.config, "num_register_tokens", 0))

    def dinov3_forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, output_hidden_states=True)
        image_features = outputs.last_hidden_state[:, self._unused_tokens :]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov3_forward(x)
