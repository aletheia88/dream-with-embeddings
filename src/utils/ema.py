from collections import OrderedDict
from typing import Callable, Optional

import torch


def update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float = 0.9999,
    param_filter: Optional[Callable[[str, torch.nn.Parameter], bool]] = None,
) -> None:
    """
    Step the EMA model towards the current model parameters.

    param_filter allows skipping parameters (e.g., positional embeddings) when desired.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param_filter is not None and not param_filter(name, param):
            continue
        if name not in ema_params:
            continue
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
