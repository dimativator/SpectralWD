"""Shared building blocks for post-training low-rank compression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.compress import LowRankLinear, effective_rank


@dataclass(frozen=True)
class LinearRef:
    name: str
    parent: nn.Module
    child_name: str
    module: nn.Linear


def target_linears(model: nn.Module, suffixes: tuple[str, ...]) -> list[LinearRef]:
    """Return stable references to eligible dense Linear modules."""
    modules = dict(model.named_modules())
    refs: list[LinearRef] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not any(name.endswith(s) for s in suffixes):
            continue
        parent_name, _, child_name = name.rpartition(".")
        refs.append(LinearRef(name, modules[parent_name] if parent_name else model, child_name, module))
    return refs


def resolve_rank(weight: torch.Tensor, rank) -> int:
    maximum = min(weight.shape)
    value = round(effective_rank(weight)) if rank == "auto" else int(rank)
    if rank == "auto":
        # A two-factor representation only compresses when r(m+n) < mn.
        out_features, in_features = weight.shape
        compressible = max(1, (out_features * in_features - 1) // (out_features + in_features))
        value = min(value, compressible)
    return max(1, min(value, maximum))


def low_rank_from_factors(layer: nn.Linear, left: torch.Tensor, right: torch.Tensor) -> LowRankLinear:
    """Create a deployable Linear ``left @ right`` factorization."""
    rank = left.shape[1]
    result = LowRankLinear(
        layer.in_features,
        layer.out_features,
        rank,
        bias=layer.bias,
    ).to(device=layer.weight.device, dtype=layer.weight.dtype)
    with torch.no_grad():
        result.A.weight.copy_(left.to(device=layer.weight.device, dtype=layer.weight.dtype))
        result.B.weight.copy_(right.to(device=layer.weight.device, dtype=layer.weight.dtype))
        if layer.bias is not None:
            result.A.bias.copy_(layer.bias)
    return result


def truncated_factors(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return balanced factors whose product is the rank-k truncated SVD."""
    u, singular, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    rank = max(1, min(rank, singular.numel()))
    root = singular[:rank].sqrt()
    return u[:, :rank] * root, root[:, None] * vh[:rank]


def replace_with_factors(ref: LinearRef, left: torch.Tensor, right: torch.Tensor) -> None:
    setattr(ref.parent, ref.child_name, low_rank_from_factors(ref.module, left, right))


def forward_loss(model: nn.Module, batches, device: str, max_batches: int | None = None) -> float:
    """Mean next-token loss on deterministic calibration batches."""
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(batches):
            if max_batches is not None and index >= max_batches:
                break
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            tokens = batch.to(device)
            output = model(
                tokens[:, :-1].contiguous(),
                targets=tokens[:, 1:].contiguous(),
            )
            losses.append(float(output["loss"].detach().float().cpu()))
    if not losses:
        raise ValueError("Calibration data is empty")
    return sum(losses) / len(losses)
