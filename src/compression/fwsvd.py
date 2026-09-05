"""FWSVD: Fisher-Weighted SVD compression.

Inspired by: https://github.com/RahulSChand/Weighted-low-rank-factorization-Pytorch

FWSVD weights each element of a weight matrix by (an approximation of) its
Fisher information before performing SVD, ensuring that directions with high
Fisher information are preserved more faithfully.  The Fisher information is
approximated by the squared gradient (diagonal Fisher approximation) averaged
over a small calibration dataset.

Typical usage:
    fisher = collect_fisher_stats(model, dataloader, n_batches=16)
    model = apply_fwsvd(model, rank=64, fisher_stats=fisher)
"""

import torch
import torch.nn.functional as F


def collect_fisher_stats(
    model: torch.nn.Module,
    dataloader,
    n_batches: int = 16,
    device: str = "cuda",
    criterion=None,
) -> dict:
    """Collect diagonal Fisher information (mean squared gradient) per linear weight.

    Args:
        model: the model to profile.
        dataloader: iterable of integer token batches, shape (B, T).
        n_batches: how many batches to average over.
        device: device to run forward passes on.
        criterion: optional callable(model, batch) -> scalar loss.  If None,
            uses standard next-token cross-entropy (assumes causal LM).

    Returns:
        dict mapping "<module_name>.weight" -> tensor of same shape as the weight.
    """
    stats: dict = {}
    model.train()

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)
        model.zero_grad()

        if criterion is not None:
            loss = criterion(model, batch)
        else:
            # Causal LM: predict next token
            out = model(batch[:, :-1])
            target = batch[:, 1:].reshape(-1)
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), target)

        loss.backward()

        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if module.weight.grad is None:
                continue
            key = name + ".weight"
            g = module.weight.grad.detach().float() ** 2
            if key not in stats:
                stats[key] = g
            else:
                stats[key] += g

    model.zero_grad()

    for k in stats:
        stats[k] = stats[k] / n_batches

    return stats


def apply_fwsvd(
    model: torch.nn.Module,
    rank: int,
    fisher_stats: dict,
    target_modules=("q_proj", "v_proj"),
    device: str = "cpu",
) -> torch.nn.Module:
    """FWSVD: Fisher-weighted SVD compression applied in-place.

    Args:
        model: the model to compress.
        rank: number of singular values to retain.
        fisher_stats: dict from collect_fisher_stats mapping
            "<module_name>.weight" to a tensor of the same shape as the weight.
        target_modules: tuple of module-name suffixes to compress.
        device: device string.

    Returns:
        The same model with compressed weights.
    """
    compressed = 0
    for name, module in model.named_modules():
        if not any(name.endswith(t) for t in target_modules):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue

        W = module.weight.data.float()
        key = name + ".weight"

        if key in fisher_stats:
            F_mat = fisher_stats[key].float().to(W.device).sqrt()
            W_weighted = W * F_mat
        else:
            W_weighted = W
            F_mat = None

        U, S, Vh = torch.linalg.svd(W_weighted, full_matrices=False)
        r = min(rank, S.numel())
        W_approx = (U[:, :r] * S[:r]) @ Vh[:r, :]

        if F_mat is not None:
            W_approx = W_approx / (F_mat + 1e-8)

        module.weight.data = W_approx.to(module.weight.dtype)
        compressed += 1

    print(f"[FWSVD] Compressed {compressed} modules with rank={rank}.")
    return model
