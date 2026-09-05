"""Activation-aware SVD (ASVD) with deployable low-rank factors.

Implements the activation-aware scaling from ASVD4LLM. Input-channel scales
are measured on calibration data, folded into the weight before truncation,
and analytically removed from the right factor afterwards.

Reference implementation: https://github.com/hahnyuan/ASVD4LLM
"""

from __future__ import annotations

import torch

from compression.common import replace_with_factors, resolve_rank, target_linears


@torch.no_grad()
def collect_activation_stats(model, dataloader, n_batches: int = 16, device: str = "cuda") -> dict:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        def hook(_module, inputs, _output, key=name + ".weight"):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            value = x.abs().sum(dim=0).cpu()
            sums[key] = sums.get(key, torch.zeros_like(value)) + value
            counts[key] = counts.get(key, 0) + x.shape[0]

        hooks.append(module.register_forward_hook(hook))
    model.eval()
    for index, batch in enumerate(dataloader):
        if index >= n_batches:
            break
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        model(batch.to(device))
    for hook in hooks:
        hook.remove()
    return {key: value / counts[key] for key, value in sums.items()}


@torch.no_grad()
def apply_asvd(
    model,
    rank,
    activation_stats: dict,
    alpha: float = 0.5,
    target_modules=("q_proj", "v_proj"),
    device: str = "cpu",
):
    comp_info: dict[str, int] = {}
    for ref in target_linears(model, tuple(target_modules)):
        weight = ref.module.weight.detach().float()
        scale = activation_stats.get(ref.name + ".weight")
        if scale is None:
            scale = torch.ones(weight.shape[1])
        scale = scale.to(weight.device).clamp_min(1e-6).pow(alpha)
        scaled = weight * scale[None, :]
        selected_rank = resolve_rank(scaled, rank)
        if selected_rank >= min(weight.shape):
            continue
        u, singular, vh = torch.linalg.svd(scaled, full_matrices=False)
        root = singular[:selected_rank].sqrt()
        left = u[:, :selected_rank] * root
        right = (root[:, None] * vh[:selected_rank]) / scale[None, :]
        replace_with_factors(ref, left, right)
        comp_info[ref.name] = selected_rank
    print(f"[ASVD] factored {len(comp_info)} modules; rank={rank}, alpha={alpha}.")
    return model, comp_info
