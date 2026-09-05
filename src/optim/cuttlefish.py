"""Cuttlefish: Low-rank training via periodic SVD projection.

Inspired by: https://github.com/hwang595/Cuttlefish

Cuttlefish trains full-rank parameters with a standard optimizer (AdamW by
default), but every `factorize_every` steps it projects each eligible 2-D
weight matrix back onto its best rank-r approximation (via truncated SVD).
This periodically enforces low-rank structure without changing the optimizer
API or requiring any model-surgery.

Eligible matrices: 2-D, both dimensions >= 10.
"""

import math

import torch
import torch.nn as nn


class Cuttlefish(torch.optim.Optimizer):
    """Wraps an inner optimizer (AdamW by default) and periodically projects
    weights to their best rank-r approximation via truncated SVD.

    Args:
        params: iterable of parameters or param groups (same as any optimizer).
        lr: learning rate passed to the inner optimizer.
        rank_fraction: fraction of min(m, n) to keep as SVD rank (0 < r <= 1).
        factorize_every: how often (in steps) to apply SVD projection.
        inner_opt_cls: optimizer class to use internally (default AdamW).
        betas: betas for the inner AdamW.
        eps: eps for the inner AdamW.
        weight_decay: weight decay for the inner AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        rank_fraction: float = 0.5,
        factorize_every: int = 100,
        inner_opt_cls=torch.optim.AdamW,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        if not 0.0 < rank_fraction <= 1.0:
            raise ValueError(f"rank_fraction must be in (0, 1], got {rank_fraction}")
        if factorize_every < 1:
            raise ValueError(f"factorize_every must be >= 1, got {factorize_every}")

        defaults = dict(
            lr=lr,
            rank_fraction=rank_fraction,
            factorize_every=factorize_every,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        # Collect param list for parent init and inner optimizer
        param_list = list(params)
        super().__init__(param_list, defaults)

        # Build the inner optimizer on the same param groups
        self._inner_opt = inner_opt_cls(
            self.param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self._step_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_eligible(p: torch.Tensor) -> bool:
        """Return True for 2-D weight matrices large enough to factorize."""
        return p.ndim == 2 and p.size(0) >= 10 and p.size(1) >= 10

    def _project_to_low_rank(self):
        """SVD-project all eligible params in-place."""
        for group in self.param_groups:
            rank_fraction = group.get("rank_fraction", self.defaults["rank_fraction"])
            for p in group["params"]:
                if not self._is_eligible(p):
                    continue
                W = p.data.float()
                m, n = W.shape
                r = max(1, int(rank_fraction * min(m, n)))
                try:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                except Exception:
                    # SVD can fail for degenerate matrices; skip silently
                    continue
                # Reconstruct best rank-r approximation
                W_lr = (U[:, :r] * S[:r]) @ Vh[:r, :]
                p.data.copy_(W_lr.to(p.dtype))

    # ------------------------------------------------------------------
    # Optimizer interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Perform the actual gradient update via the inner optimizer
        self._inner_opt.step()
        self._step_count += 1

        # Periodically project to low-rank (also on step 1 so init is low-rank)
        factorize_every = self.defaults["factorize_every"]
        if self._step_count % factorize_every == 0:
            self._project_to_low_rank()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        self._inner_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        sd = super().state_dict()
        sd["_inner_opt"] = self._inner_opt.state_dict()
        sd["_step_count"] = self._step_count
        return sd

    def load_state_dict(self, state_dict):
        inner_sd = state_dict.pop("_inner_opt", None)
        step = state_dict.pop("_step_count", 0)
        super().load_state_dict(state_dict)
        if inner_sd is not None:
            self._inner_opt.load_state_dict(inner_sd)
        self._step_count = step

    def __repr__(self):
        return (
            f"Cuttlefish(lr={self.defaults['lr']}, "
            f"rank_fraction={self.defaults['rank_fraction']}, "
            f"factorize_every={self.defaults['factorize_every']}, "
            f"inner_opt={self._inner_opt.__class__.__name__})"
        )
