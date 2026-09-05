"""SVD-LLM: Whitening-based SVD compression.

Paper / repo: https://github.com/AIoT-MLSys-Lab/SVD-LLM

SVD-LLM multiplies each weight matrix by the square-root of the input
activation covariance before performing SVD, aligning the decomposition with
the actual input distribution.  This yields a better low-rank approximation
in terms of reconstruction error weighted by the input statistics.

Typical usage:
    cov = collect_input_covariance(model, dataloader, n_batches=16)
    model = apply_svd_llm(model, rank=64, whitening_stats=cov)
"""

import torch


@torch.no_grad()
def stable_cholesky_whitening(
    covariance: torch.Tensor,
    relative_eigenvalue_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a numerically bounded Cholesky whitening factor and its inverse.

    Calibration covariances can be nearly singular in individual projection
    layers.  A fixed absolute ridge is not scale-aware and can disappear in a
    float32 eigensolve, yielding a near-infinite inverse-whitening factor.  The
    ridge below enforces a relative eigenvalue floor in float64.
    """
    if relative_eigenvalue_floor <= 0:
        raise ValueError("relative_eigenvalue_floor must be positive")

    covariance = covariance.detach().double()
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(covariance)
    spectral_scale = eigenvalues.abs().max().clamp_min(1.0)
    eigenvalue_floor = relative_eigenvalue_floor * spectral_scale
    ridge = (eigenvalue_floor - eigenvalues.min()).clamp_min(0.0)
    regularized = covariance + ridge * torch.eye(
        covariance.shape[0], device=covariance.device, dtype=covariance.dtype
    )
    whitening = torch.linalg.cholesky(regularized)
    inverse_whitening = torch.linalg.solve_triangular(
        whitening,
        torch.eye(
            whitening.shape[0], device=whitening.device, dtype=whitening.dtype
        ),
        upper=False,
    )
    return whitening, inverse_whitening


def rank_with_residual_guard(
    singular_values: torch.Tensor,
    rank: int,
    multiple: int | None,
    max_relative_residual: float,
) -> int:
    """Increase a proposed rank until whitened tail energy is bounded."""
    if not 0 < max_relative_residual < 1:
        raise ValueError("max_relative_residual must be in (0, 1)")
    rank = min(rank, singular_values.numel())
    total_energy = singular_values.square().sum()
    while rank < singular_values.numel():
        residual = (singular_values[rank:].square().sum() / total_energy).sqrt()
        if residual <= max_relative_residual:
            break
        step = multiple if multiple is not None else 1
        rank = min(rank + step, singular_values.numel())
    return rank


def collect_input_covariance(
    model: torch.nn.Module,
    dataloader,
    n_batches: int = 16,
    device: str = "cuda",
) -> dict:
    """Collect input covariance matrices for each linear layer.

    Args:
        model: the model to profile.
        dataloader: iterable of integer token batches, shape (B, T).
        n_batches: how many batches to average over.
        device: device to run forward passes on.

    Returns:
        dict mapping "<module_name>.weight" -> 2-D tensor of shape
        (in_features, in_features).
    """
    stats: dict = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        def _hook(mod, inp, out, _name=name):
            x = inp[0].detach().float()
            x_flat = x.reshape(-1, x.shape[-1])  # (N, in_features)
            cov = x_flat.T @ x_flat / x_flat.shape[0]
            key = _name + ".weight"
            if key not in stats:
                stats[key] = cov
            else:
                stats[key] += cov

        hooks.append(module.register_forward_hook(_hook))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            model(batch)

    for h in hooks:
        h.remove()

    for k in stats:
        stats[k] = stats[k] / n_batches

    return stats


def apply_svd_llm(
    model: torch.nn.Module,
    rank,
    whitening_stats: dict,
    target_modules=("q_proj", "v_proj"),
    device: str = "cpu",
    margin: int = 0,
    low_rank_kernel: str = "torch",
    auto_rank_multiple: int | None = None,
    max_whitened_relative_residual: float | None = 0.05,
) -> tuple[torch.nn.Module, dict]:
    """SVD-LLM: whitening-aware SVD compression, applied in-place.

    Each target ``nn.Linear`` is replaced by a :class:`LowRankLinear` so the
    model genuinely shrinks (W ≈ A @ B with A: (out, r), B: (r, in)).

    Args:
        model: the model to compress (modified in-place).
        rank: number of singular values to retain, or the string ``"auto"``
            to use each layer's (rounded) effective rank.
        whitening_stats: dict from collect_input_covariance mapping
            "<module_name>.weight" to (in_features, in_features) covariance.
        target_modules: tuple of module-name suffixes to compress.
        device: device string.
        margin: only used with ``rank="auto"``; retained rank becomes
            ``round(effective_rank(W)) + margin`` (may be negative to compress
            more aggressively than the bare effective rank).
        low_rank_kernel: ``"torch"`` for two ``Linear`` calls or ``"triton"``
            for the inference-only fused prototype with an automatic fallback.
        auto_rank_multiple: if set with ``rank="auto"``, floor each retained
            rank to this multiple. A zero result is clamped to rank 16, the
            smallest rank used by this BF16 tensor-core experiment. Must be a
            positive multiple of 16.
        max_whitened_relative_residual: for auto rank, raise the retained rank
            in aligned steps until the whitened SVD tail has no more than this
            relative Frobenius norm. Set to ``None`` to disable the guard.

    Returns:
        ``(model, comp_info)`` where ``comp_info`` maps each compressed layer
        name to the rank it was reduced to.
    """
    from models.compress import LowRankLinear, effective_rank

    auto = isinstance(rank, str) and rank == "auto"
    if auto_rank_multiple is not None:
        if not auto:
            raise ValueError("auto_rank_multiple requires rank='auto'")
        if auto_rank_multiple < 16 or auto_rank_multiple % 16 != 0:
            raise ValueError(
                "auto_rank_multiple must be a positive multiple of 16"
            )
    comp_info: dict[str, int] = {}

    for parent_name, parent in list(model.named_modules()):
        for child_name, module in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not any(full_name.endswith(t) for t in target_modules):
                continue
            if not isinstance(module, torch.nn.Linear):
                continue

            W = module.weight.data.float()  # (out, in)
            out_f, in_f = W.shape
            max_rank = min(out_f, in_f)

            r = round(effective_rank(W)) + margin if auto else int(rank)
            if auto_rank_multiple is not None:
                r = max(16, (r // auto_rank_multiple) * auto_rank_multiple)
            r = max(1, min(r, max_rank))
            if r >= max_rank:
                continue  # no compression possible

            key = full_name + ".weight"
            if key in whitening_stats:
                uses_whitening = True
                X_cov = whitening_stats[key].float().to(W.device)  # (in, in)
                whitening, inverse_whitening = stable_cholesky_whitening(X_cov)
                W_white = W.double() @ whitening
            else:
                uses_whitening = False
                W_white = W
                inverse_whitening = None

            U, S, Vh = torch.linalg.svd(W_white, full_matrices=False)
            if (
                auto
                and uses_whitening
                and max_whitened_relative_residual is not None
            ):
                r = rank_with_residual_guard(
                    S,
                    r,
                    auto_rank_multiple,
                    max_whitened_relative_residual,
                )
            sqrt_s = S[:r].sqrt()
            A_weight = U[:, :r] * sqrt_s.unsqueeze(0)  # (out, r)
            B_weight = sqrt_s.unsqueeze(1) * Vh[:r, :] # (r, in)

            if inverse_whitening is not None:
                # Undo whitening on the right factor: W ≈ A @ (B @ inv_sqrt_cov)
                B_weight = B_weight @ inverse_whitening

            bias = module.bias.data if module.bias is not None else None
            low_rank = LowRankLinear(
                in_f,
                out_f,
                r,
                bias=bias,
                kernel=low_rank_kernel,
            ).to(W.device)
            with torch.no_grad():
                low_rank.A.weight.copy_(A_weight.to(module.weight.dtype))
                low_rank.B.weight.copy_(B_weight.to(module.weight.dtype))
            setattr(parent, child_name, low_rank)
            comp_info[full_name] = r

    tag = "auto" if auto else rank
    if auto_rank_multiple is not None:
        tag = f"{tag}, floor-multiple={auto_rank_multiple}"
    print(f"[SVD-LLM] Compressed {len(comp_info)} modules with rank={tag}.")
    return model, comp_info
