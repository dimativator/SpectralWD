import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Approximates a Linear layer as two smaller linear layers via truncated SVD.

    W ≈ U_r @ diag(S_r) @ Vh_r  is stored as:
        A: (out_features, rank)  — weight = U_r @ diag(S_r)
        B: (rank, in_features)   — weight = Vh_r
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias=None,
        kernel: str = "torch",
    ):
        super().__init__()
        if kernel not in ("torch", "triton"):
            raise ValueError(f"Unknown low-rank kernel: {kernel}")
        self.B = nn.Linear(in_features, rank, bias=False)
        self.A = nn.Linear(rank, out_features, bias=bias is not None)
        self.kernel = kernel
        if bias is not None:
            with torch.no_grad():
                self.A.bias.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == "triton":
            from models.fused_low_rank import fused_low_rank_linear

            return fused_low_rank_linear(
                x,
                self.B.weight,
                self.A.weight,
                self.A.bias,
            )
        return self.A(self.B(x))


def _svd_compress_linear(layer: nn.Linear, rank: int) -> LowRankLinear:
    W = layer.weight.data.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # U: (out, k), S: (k,), Vh: (k, in)  where k = min(out, in)
    rank = min(rank, S.shape[0])

    A_weight = U[:, :rank] * S[:rank].unsqueeze(0)  # (out, rank)
    B_weight = Vh[:rank, :]                          # (rank, in)

    bias = layer.bias.data if layer.bias is not None else None
    device = layer.weight.device
    low_rank = LowRankLinear(
        layer.in_features, layer.out_features, rank, bias=bias
    ).to(device)
    with torch.no_grad():
        low_rank.A.weight.copy_(A_weight.to(device=device, dtype=layer.weight.dtype))
        low_rank.B.weight.copy_(B_weight.to(device=device, dtype=layer.weight.dtype))

    return low_rank


def compress_model_svd(
    model: nn.Module,
    rank: int,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
    only_names: tuple[str, ...] | None = None,
) -> nn.Module:
    """Replace every nn.Linear in *model* with a LowRankLinear approximation.

    Layers whose name contains any string from *skip_names* are left intact
    (embedding / unembedding layers are typically kept full-rank). If
    *only_names* is given, only layers whose name ends with one of those
    suffixes are compressed (everything else is left intact).
    Modifies the model in-place and returns it.
    """
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue

            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(skip in full_name for skip in skip_names):
                continue
            if only_names and not any(full_name.endswith(t) for t in only_names):
                continue

            in_f, out_f = child_module.in_features, child_module.out_features
            effective_rank = min(rank, in_f, out_f)
            if effective_rank >= min(in_f, out_f):
                # no compression possible
                continue

            low_rank = _svd_compress_linear(child_module, effective_rank)
            setattr(parent_module, child_name, low_rank)

    return model


@torch.no_grad()
def compress_model_svd_adaptive(
    model: nn.Module,
    r_gap: int = 0,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
    only_names: tuple[str, ...] | None = None,
) -> nn.Module:
    """Replace every nn.Linear with a LowRankLinear compressed to its own effective rank.

    The target rank for each layer is round(effective_rank(W)) + r_gap, so r_gap
    lets you compress slightly more (negative) or less (positive) than the bare
    effective rank. If *only_names* is given, only layers whose name ends with
    one of those suffixes are compressed. Modifies the model in-place and
    returns it.
    """
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue

            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(skip in full_name for skip in skip_names):
                continue
            if only_names and not any(full_name.endswith(t) for t in only_names):
                continue

            in_f, out_f = child_module.in_features, child_module.out_features
            max_rank = min(in_f, out_f)

            er = effective_rank(child_module.weight.data)
            target_rank = min(max(round(er) + r_gap, 1), max_rank)
            if target_rank >= max_rank:
                continue

            low_rank = _svd_compress_linear(child_module, target_rank)
            setattr(parent_module, child_name, low_rank)
            print(f"  {full_name}: erank={er:.1f} → rank={target_rank} (max={max_rank})")

    return model


@torch.no_grad()
def effective_rank(W: torch.Tensor) -> float:
    """Entropy-based effective rank (Roy & Vetterli, 2007).

    erank(W) = exp( -sum_i p_i * log(p_i) )  where p_i = s_i / sum(s).
    Ranges from 1 (rank-1 matrix) to min(m, n) (uniform singular spectrum).
    """
    s = torch.linalg.svdvals(W.float())
    s = s[s > 0]
    p = s / s.sum()
    return torch.exp(-(p * p.log()).sum()).item()


@torch.no_grad()
def stable_rank(W: torch.Tensor) -> float:
    """Stable rank of a matrix.

    srank(W) = ||W||_F^2 / ||W||_2^2 = sum_i sigma_i^2 / sigma_1^2.
    Returns 0 for the all-zero matrix. For nonzero matrices, stable rank ranges
    from 1 to rank(W).
    """
    s = torch.linalg.svdvals(W.float())
    spectral_norm_sq = s[0].square()
    if spectral_norm_sq == 0:
        return 0.0
    return (s.square().sum() / spectral_norm_sq).item()


@torch.no_grad()
def compress_embeddings_inplace(
    model: nn.Module,
    rank,
    only_names: tuple[str, ...],
) -> dict[str, int]:
    """Replace token-embedding matrices with a rank-r SVD approximation in place.

    Only ``nn.Embedding`` modules whose name ends with one of *only_names* are
    touched. The (vocab x n_embd) weight is overwritten by its rank-r SVD
    reconstruction (dense, same shape) rather than restructured into a factored
    module — this keeps tying intact: when ``wte.weight is lm_head.weight``
    (the default), overwriting ``weight.data`` updates the output head too, so a
    single shared low-rank matrix is used for both embedding lookup and logits.

    ``rank`` may be an int or the string ``"auto"`` (per-matrix effective rank).
    Returns ``{name: retained_rank}`` for each embedding actually reduced, so the
    caller can fold it into its compression-rate accounting.
    """
    auto = isinstance(rank, str) and rank == "auto"
    info: dict[str, int] = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Embedding):
            continue
        if not any(name.endswith(t) for t in only_names):
            continue

        W = m.weight.data.float()  # (vocab, n_embd)
        max_rank = min(W.shape)
        r = round(effective_rank(W)) if auto else int(rank)
        r = max(1, min(r, max_rank))
        if r >= max_rank:
            continue  # no compression possible

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        W_approx = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
        m.weight.data = W_approx.to(m.weight.dtype)
        info[name] = r

    return info


def _weight_as_2d(module: nn.Module) -> torch.Tensor:
    """Linear weights are already 2D; Conv2d weights [out_ch, in_ch, kh, kw]
    are viewed as the standard "filter matrix" [out_ch, in_ch*kh*kw], matching
    the reshape AdamWSpectralL1Reg applies before its nuclear-norm prox."""
    W = module.weight.data
    return W.view(W.shape[0], -1) if isinstance(module, nn.Conv2d) else W


@torch.no_grad()
def model_effective_ranks(
    model: nn.Module,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
) -> dict[str, float]:
    """Compute effective rank for every Linear/Conv2d weight matrix in *model*
    (Conv2d weights viewed as a 2D filter matrix, see `_weight_as_2d`).

    Returns a dict with:
      - ``"effective_rank/<layer_name>"`` for each qualifying layer
      - ``"effective_rank/mean_weighted"`` — weighted average across layers,
        weights = min(out, in) (number of singular values per layer)
    """
    ranks: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if any(skip in name for skip in skip_names):
            continue
        W = _weight_as_2d(module)
        er = effective_rank(W)
        ranks[f"effective_rank/{name}"] = er
        k = min(W.shape)
        weighted_sum += er * k
        total_weight += k

    if total_weight > 0:
        ranks["effective_rank/mean_weighted"] = weighted_sum / total_weight

    return ranks


@torch.no_grad()
def model_stable_ranks(
    model: nn.Module,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
) -> dict[str, float]:
    """Compute stable rank for every Linear/Conv2d weight matrix in *model*.

    Conv2d weights are viewed as 2D filter matrices, matching
    :func:`model_effective_ranks`. Returns a dict with:
      - ``"stable_rank/<layer_name>"`` for each qualifying layer
      - ``"stable_rank/mean_weighted"`` — weighted average across layers,
        weights = min(out, in) (number of singular values per layer)
    """
    ranks: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if any(skip in name for skip in skip_names):
            continue
        W = _weight_as_2d(module)
        sr = stable_rank(W)
        ranks[f"stable_rank/{name}"] = sr
        k = min(W.shape)
        weighted_sum += sr * k
        total_weight += k

    if total_weight > 0:
        ranks["stable_rank/mean_weighted"] = weighted_sum / total_weight

    return ranks


@torch.no_grad()
def grad_effective_ranks(
    model: nn.Module,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
) -> dict[str, float]:
    """Weighted-mean effective rank of 2-D gradient tensors."""
    weighted_sum = 0.0
    total_weight = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name for skip in skip_names):
            continue
        if module.weight.grad is None:
            continue
        er = effective_rank(module.weight.grad)
        k = min(module.weight.shape)
        weighted_sum += er * k
        total_weight += k
    result: dict[str, float] = {}
    if total_weight > 0:
        result["grad_effective_rank/mean_weighted"] = weighted_sum / total_weight
    return result


@torch.no_grad()
def optimizer_state_effective_ranks(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "wpe"),
) -> dict[str, float]:
    """Weighted-mean effective rank of 2-D optimizer state tensors.

    Auto-detects all 2-D Tensor state entries (e.g. first_momentum, second_momentum).
    """
    param_to_name = {id(p): name for name, p in model.named_parameters()}
    sums: dict[str, float] = {}
    totals: dict[str, int] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.dim() != 2:
                continue
            name = param_to_name.get(id(p), "")
            if any(skip in name for skip in skip_names):
                continue
            state = optimizer.state[p]
            if not state:
                continue
            k = min(p.shape)
            for key, val in state.items():
                if not isinstance(val, torch.Tensor) or val.dim() != 2:
                    continue
                er = effective_rank(val)
                sums[key] = sums.get(key, 0.0) + er * k
                totals[key] = totals.get(key, 0) + k
    result: dict[str, float] = {}
    for key in sums:
        if totals[key] > 0:
            result[f"opt_{key}_effective_rank/mean_weighted"] = sums[key] / totals[key]
    return result


def compression_stats(original: nn.Module, compressed: nn.Module) -> dict:
    orig_params = sum(p.numel() for p in original.parameters())
    comp_params = sum(p.numel() for p in compressed.parameters())
    return {
        "original_params": orig_params,
        "compressed_params": comp_params,
        "compression_ratio": orig_params / comp_params if comp_params > 0 else float("inf"),
    }
