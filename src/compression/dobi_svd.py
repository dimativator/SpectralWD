"""Dobi-SVD: differentiable activation-rank allocation and IPCA update.

This implements the non-remapped Dobi-SVD path. Continuous truncation
positions are optimized under a factor-parameter budget. Final weights are
updated by projecting onto principal output-activation directions, matching
the IPCA reconstruction derived in the official implementation.

Reference implementation: https://github.com/wangqinsi1/Dobi-SVD
"""

from __future__ import annotations

import torch

from compression.common import replace_with_factors, resolve_rank, target_linears


@torch.no_grad()
def collect_output_covariances(
    model,
    dataloader,
    target_modules,
    n_batches: int = 16,
    device: str = "cuda",
    max_tokens: int = 2048,
):
    refs = target_linears(model, tuple(target_modules))
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []
    for ref in refs:
        def hook(_module, _inputs, output, name=ref.name):
            x = output.detach().float().reshape(-1, output.shape[-1])
            if x.shape[0] > max_tokens:
                stride = max(1, x.shape[0] // max_tokens)
                x = x[::stride][:max_tokens]
            covariance = (x.T @ x).cpu()
            sums[name] = sums.get(name, torch.zeros_like(covariance)) + covariance
            counts[name] = counts.get(name, 0) + x.shape[0]
        hooks.append(ref.module.register_forward_hook(hook))
    model.eval()
    for index, batch in enumerate(dataloader):
        if index >= n_batches:
            break
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        model(batch.to(device))
    for hook in hooks:
        hook.remove()
    return {name: sums[name] / counts[name] for name in sums}


def _differentiate_ranks(refs, spectra, target_budget: float, steps: int, device: str) -> dict[str, int]:
    maxima = torch.tensor([min(ref.module.weight.shape) for ref in refs], device=device, dtype=torch.float32)
    costs = torch.tensor(
        [sum(ref.module.weight.shape) for ref in refs], device=device, dtype=torch.float32
    )
    dense_budget = sum(ref.module.weight.numel() for ref in refs)
    initial_ratio = min(1.0, target_budget / dense_budget)
    gamma = torch.nn.Parameter((maxima * initial_ratio).clamp_min(1))
    optimizer = torch.optim.Adam([gamma], lr=0.1)
    indices = [torch.arange(1, spectrum.numel() + 1, device=device) for spectrum in spectra]
    normalized = [spectrum.to(device).clamp_min(0) / spectrum.sum().clamp_min(1e-12) for spectrum in spectra]
    for _ in range(steps):
        optimizer.zero_grad()
        reconstruction = torch.zeros((), device=device)
        for layer_index, (spectrum, index) in enumerate(zip(normalized, indices)):
            gate = 0.5 * torch.tanh(4.0 * (gamma[layer_index] - index)) + 0.5
            reconstruction = reconstruction + ((1.0 - gate).square() * spectrum).sum()
        used = (gamma.clamp(min=1) * costs).sum()
        budget_error = (used - target_budget) / max(target_budget, 1.0)
        loss = reconstruction / len(refs) + 20.0 * budget_error.square()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            gamma.clamp_(min=1)
            gamma.copy_(torch.minimum(gamma, maxima))
    return {ref.name: int(round(value)) for ref, value in zip(refs, gamma.detach().cpu().tolist())}


def apply_dobi_svd(
    model,
    rank,
    output_covariances: dict,
    target_modules=("q_proj", "v_proj"),
    device: str = "cuda",
    allocation_steps: int = 300,
):
    refs = [ref for ref in target_linears(model, tuple(target_modules)) if ref.name in output_covariances]
    if not refs:
        raise ValueError("Dobi-SVD found no calibrated target modules")
    eigensystems = []
    for ref in refs:
        values, vectors = torch.linalg.eigh(output_covariances[ref.name].double())
        maximum = min(ref.module.weight.shape)
        eigensystems.append((values[-maximum:].flip(0).float(), vectors[:, -maximum:].flip(1).float()))
    if rank == "auto":
        target_budget = float(sum(
            resolve_rank(ref.module.weight.detach().float(), "auto") * sum(ref.module.weight.shape)
            for ref in refs
        ))
        ranks = _differentiate_ranks(
            refs,
            [values for values, _ in eigensystems],
            target_budget,
            allocation_steps,
            device,
        )
    else:
        ranks = {ref.name: resolve_rank(ref.module.weight.detach().float(), rank) for ref in refs}
    comp_info: dict[str, int] = {}
    with torch.no_grad():
        for ref, (_values, vectors) in zip(refs, eigensystems):
            selected_rank = max(1, min(ranks[ref.name], vectors.shape[1]))
            if selected_rank >= min(ref.module.weight.shape):
                continue
            left = vectors[:, :selected_rank].to(ref.module.weight.device)
            right = left.T @ ref.module.weight.detach().float()
            replace_with_factors(ref, left, right)
            comp_info[ref.name] = selected_rank
    print(f"[Dobi-SVD] IPCA-updated {len(comp_info)} modules; rank={rank}, remapping=False.")
    return model, comp_info
