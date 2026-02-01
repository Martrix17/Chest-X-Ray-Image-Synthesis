"""
EMA class for evaluation.
"""

import copy
from typing import Any, Dict, Self

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average (EMA) wrapper for model parameters.

    Keeps a shadow copy of model weights:
        ema = decay * ema + (1 - decay) * model
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: Module to copy.
            decay: Decay rate.
        """
        device = next(model.parameters()).device
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        self.shadow.requires_grad_(False)
        self.shadow.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA parameters from model.
        """
        for ema_param, model_param in zip(
            self.shadow.parameters(),
            model.parameters(),
            strict=True,
        ):
            ema_param.data.mul_(self.decay)
            ema_param.data.add_(model_param.data, alpha=1.0 - self.decay)

    def to(self, device: torch.device) -> Self:
        """Move shadow model to device."""
        self.shadow = self.shadow.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Get shadow model state dict."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict) -> None:
        """Load shadow model state dict."""
        self.shadow.load_state_dict(state_dict)

    def get_model(self) -> nn.Module:
        """Get the shadow model for inference."""
        return self.shadow
