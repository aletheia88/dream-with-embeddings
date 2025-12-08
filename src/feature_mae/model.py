import math
from typing import Tuple

import torch
from torch import nn


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Initialize a tensor with values drawn from a truncated normal distribution."""
    return nn.init.trunc_normal_(tensor, mean=mean, std=std)


class FeatureMaskedAutoencoder(nn.Module):
    """MAE-style autoencoder that operates on representation tokens instead of image patches."""

    def __init__(
        self,
        *,
        feature_dim: int,
        num_tokens: int,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens

        if encoder_embed_dim != feature_dim:
            raise ValueError(
                f"encoder_embed_dim ({encoder_embed_dim}) must match feature_dim ({feature_dim}) when no projection is used."
            )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_embed_dim,
            nhead=encoder_num_heads,
            dim_feedforward=int(encoder_embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, encoder_embed_dim))
        self.encoder_cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, decoder_embed_dim))
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, feature_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        trunc_normal_(self.encoder_pos_embed, std=0.02)
        trunc_normal_(self.decoder_pos_embed, std=0.02)
        trunc_normal_(self.encoder_cls_token, std=0.02)
        trunc_normal_(self.decoder_cls_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)

        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.xavier_uniform_(self.decoder_pred.weight)

        if self.decoder_embed.bias is not None:
            nn.init.zeros_(self.decoder_embed.bias)
        if self.decoder_pred.bias is not None:
            nn.init.zeros_(self.decoder_pred.bias)

    @staticmethod
    def _random_mask(
        x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly masks tokens.

        Returns the partially observed tokens, the binary mask, and indices to restore order.
        """
        num_tokens = x.shape[1]
        len_keep = int(round(num_tokens * (1 - mask_ratio)))

        noise = torch.rand(x.size(0), num_tokens, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        gather_idx = ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2])
        x_masked = torch.gather(x, dim=1, index=gather_idx)

        mask = torch.ones(x.size(0), num_tokens, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, features: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes visible tokens with random masking."""
        x = features
        x = x + self.encoder_pos_embed[:, 1:, :]

        x_visible, mask, ids_restore = self._random_mask(x, mask_ratio)

        cls_tokens = self.encoder_cls_token + self.encoder_pos_embed[:, :1, :]
        cls_tokens = cls_tokens.expand(features.size(0), -1, -1)
        x = torch.cat((cls_tokens, x_visible), dim=1)

        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Decodes full sequence from visible tokens and mask tokens."""
        x = self.decoder_embed(x)

        cls_token = self.decoder_cls_token + self.decoder_pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.size(0), -1, -1)

        x_without_cls = x[:, 1:, :]

        mask_tokens = self.mask_token.repeat(
            x.size(0), ids_restore.size(1) - x_without_cls.size(1), 1
        )
        x_ = torch.cat([x_without_cls, mask_tokens], dim=1)
        index = ids_restore.unsqueeze(-1).expand(-1, -1, x_.size(-1))
        x_ = torch.gather(x_, dim=1, index=index)

        x = torch.cat([cls_token, x_], dim=1)
        x = x + self.decoder_pos_embed

        x = self.decoder(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)

    def forward(
        self,
        features: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_output, mask, ids_restore = self.forward_encoder(features, mask_ratio)
        pred = self.forward_decoder(encoder_output, ids_restore)
        pred = pred[:, 1:, :]

        target = features
        return pred, target, mask

    def loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
