from typing import Dict, Type

import torch.nn as nn

from .config import MLPLabelNoiseConfig

FACTORY: Dict[str, Type[nn.Module]] = {}


def register(name):
    def decorator(cls):
        FACTORY[name] = cls
        return cls

    return decorator


@register("mlp")
class MLP(nn.Module):
    """Fully-connected classifier: every weight is a 2D nn.Linear matrix, so
    spectral WD (AdamWSpectralL1Reg) applies to all of them, same as the toy
    study's 3-layer MLP -- just deeper/wider here."""

    def __init__(self, cfg: MLPLabelNoiseConfig, in_dim: int = 28 * 28, out_dim: int = 10):
        super().__init__()
        dims = [in_dim] + [cfg.hidden_dim] * cfg.n_hidden_layers
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.head(self.hidden(x))


def get_model(name: str, cfg: MLPLabelNoiseConfig) -> nn.Module:
    if name not in FACTORY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(FACTORY)}")
    return FACTORY[name](cfg)
