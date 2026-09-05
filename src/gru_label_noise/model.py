import torch
import torch.nn as nn

from .config import GRULabelNoiseConfig


class RowSequentialGRU(nn.Module):
    """Reads an MNIST image as 28 consecutive row vectors of length 28."""

    def __init__(self, cfg: GRULabelNoiseConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=28,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.hidden_dim, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        rows = images.squeeze(1)
        _, hidden = self.gru(rows)
        return self.head(hidden[-1])
