"""SliceGPT for the native llm-baselines LLaMA implementation.

The implementation performs the defining SliceGPT operations: calibration
PCA of residual-stream signals, RMSNorm fusion, orthogonal basis changes,
architectural hidden-dimension slicing, and explicit residual shortcut maps.
It intentionally supports only the native ``models.llama.Llama`` layout so an
unsupported architecture cannot silently fall back to ordinary weight SVD.

Reference implementation: https://github.com/microsoft/TransformerCompression
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llama import RMSNorm, apply_rotary_emb


def _linear(weight: torch.Tensor, bias=None) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    layer = layer.to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(bias)
    return layer


class _SlicedAttention(nn.Module):
    def __init__(self, original, input_basis: torch.Tensor, output_basis: torch.Tensor, norm_weight):
        super().__init__()
        hidden = original.n_embd
        self.n_embd = hidden
        self.n_head = original.n_head
        self.flash = original.flash
        self.dropout = original.dropout
        self.attn_dropout = copy.deepcopy(original.attn_dropout)
        self.resid_dropout = copy.deepcopy(original.resid_dropout)
        if hasattr(original, "bias"):
            self.register_buffer("bias", original.bias.detach().clone(), persistent=False)
        fused_input = original.c_attn.weight.detach() * norm_weight[None, :]
        self.c_attn = _linear(fused_input @ input_basis)
        self.c_proj = _linear(output_basis.T @ original.c_proj.weight.detach())

    def forward(self, x, freqs_cis):
        batch, length, _ = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = self.n_embd // self.n_head
        q = q.view(batch, length, self.n_head, head_dim)
        k = k.view(batch, length, self.n_head, head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            att = att.masked_fill(self.bias[:, :, :length, :length] == 0, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, length, self.n_embd)
        return self.resid_dropout(self.c_proj(y))


class _SlicedMLP(nn.Module):
    def __init__(self, original, input_basis: torch.Tensor, output_basis: torch.Tensor, norm_weight):
        super().__init__()
        scale = norm_weight[None, :]
        self.w1 = _linear((original.w1.weight.detach() * scale) @ input_basis)
        self.w2 = _linear((original.w2.weight.detach() * scale) @ input_basis)
        self.c_proj = _linear(output_basis.T @ original.c_proj.weight.detach())

    def forward(self, x):
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x)), {}


class _SlicedBlock(nn.Module):
    def __init__(self, original, input_basis, middle_basis, output_basis):
        super().__init__()
        self.ln_1 = RMSNorm(input_basis.shape[1], eps=original.ln_1.eps).to(
            device=input_basis.device, dtype=input_basis.dtype
        )
        self.attn = _SlicedAttention(original.attn, input_basis, middle_basis, original.ln_1.weight.detach())
        self.ln_2 = RMSNorm(middle_basis.shape[1], eps=original.ln_2.eps).to(
            device=input_basis.device, dtype=input_basis.dtype
        )
        if not all(hasattr(original.mlp, name) for name in ("w1", "w2", "c_proj")):
            raise NotImplementedError("SliceGPT does not support MoE blocks")
        self.mlp = _SlicedMLP(original.mlp, middle_basis, output_basis, original.ln_2.weight.detach())
        self.register_buffer("attn_shortcut", middle_basis.T @ input_basis)
        self.register_buffer("mlp_shortcut", output_basis.T @ middle_basis)

    def forward(self, x, freqs_cis):
        x = F.linear(x, self.attn_shortcut) + self.attn(self.ln_1(x), freqs_cis)
        mlp_out, aux = self.mlp(self.ln_2(x))
        return F.linear(x, self.mlp_shortcut) + mlp_out, aux


@torch.no_grad()
def _collect_residual_covariances(model, dataloader, n_batches: int, device: str, max_tokens: int = 2048):
    blocks = list(model.transformer.h)
    points = [block.ln_1 for block in blocks] + [block.ln_2 for block in blocks] + [model.transformer.ln_f]
    sums = [None] * len(points)
    counts = [0] * len(points)
    hooks = []
    for point_index, module in enumerate(points):
        def hook(_module, inputs, _output, index=point_index):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            if x.shape[0] > max_tokens:
                stride = max(1, x.shape[0] // max_tokens)
                x = x[::stride][:max_tokens]
            covariance = x.T @ x
            sums[index] = covariance.cpu() if sums[index] is None else sums[index] + covariance.cpu()
            counts[index] += x.shape[0]
        hooks.append(module.register_forward_hook(hook))
    model.eval()
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= n_batches:
            break
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        model(batch.to(device))
    for hook in hooks:
        hook.remove()
    if any(value is None for value in sums):
        raise RuntimeError("SliceGPT failed to collect every residual-stream signal")
    return [value / count for value, count in zip(sums, counts)]


def _activation_effective_rank(covariance: torch.Tensor) -> float:
    values = torch.linalg.eigvalsh(covariance.double()).clamp_min(0).sqrt()
    values = values[values > 0]
    probabilities = values / values.sum()
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


@torch.no_grad()
def apply_slice_gpt(
    model,
    slice_fraction: float = 0.1,
    target_modules=None,
    n_batches_pca: int = 8,
    dataloader=None,
    device: str = "cuda",
    rank=None,
):
    if dataloader is None:
        raise ValueError("SliceGPT requires calibration data")
    if not all(hasattr(model, name) for name in ("transformer", "lm_head")):
        raise NotImplementedError("SliceGPT currently supports only native llm-baselines LLaMA")
    blocks = list(model.transformer.h)
    hidden = model.config.n_embd
    covariances = _collect_residual_covariances(model, dataloader, n_batches_pca, device)
    if rank == "auto":
        retained = round(sum(_activation_effective_rank(cov) for cov in covariances) / len(covariances))
    elif rank is not None:
        retained = int(rank)
    else:
        retained = round(hidden * (1.0 - slice_fraction))
    retained = max(8, min(hidden, retained))
    retained -= retained % 8
    retained = max(8, retained)
    # The native checkpoints tie embeddings and the LM head. SliceGPT needs
    # different input/output PCA bases and therefore unties them. Cap the
    # automatic width so this architectural conversion is still a compression.
    if rank == "auto":
        original_parameters = sum(parameter.numel() for parameter in model.parameters())
        vocab = model.transformer.wte.num_embeddings
        per_dimension = 2 * vocab + 1
        for block in blocks:
            intermediate = block.mlp.w1.out_features
            per_dimension += 4 * hidden + 3 * intermediate + 2
        nonexpanding = max(8, (original_parameters - 1) // per_dimension)
        nonexpanding -= nonexpanding % 8
        retained = min(retained, nonexpanding)
    if retained == hidden:
        return model, {}

    bases = []
    for covariance in covariances:
        _, vectors = torch.linalg.eigh(covariance.double())
        bases.append(vectors[:, -retained:].to(device=device, dtype=next(model.parameters()).dtype))
    count = len(blocks)
    input_bases = bases[:count]
    middle_bases = bases[count : 2 * count]
    final_basis = bases[-1]
    output_bases = input_bases[1:] + [final_basis]

    embedding = model.transformer.wte
    new_embedding = nn.Embedding(embedding.num_embeddings, retained).to(
        device=embedding.weight.device, dtype=embedding.weight.dtype
    )
    new_embedding.weight.copy_(embedding.weight.detach() @ input_bases[0])
    new_blocks = nn.ModuleList(
        [_SlicedBlock(block, q_in, q_mid, q_out) for block, q_in, q_mid, q_out in zip(
            blocks, input_bases, middle_bases, output_bases
        )]
    )
    final_norm = model.transformer.ln_f
    fused_head = model.lm_head.weight.detach() * final_norm.weight.detach()[None, :]
    model.transformer.wte = new_embedding
    model.transformer.h = new_blocks
    model.transformer.ln_f = RMSNorm(retained, eps=final_norm.eps).to(
        device=final_norm.weight.device, dtype=final_norm.weight.dtype
    )
    model.lm_head = _linear(fused_head @ final_basis)
    model.slice_gpt_hidden_size = retained
    print(f"[SliceGPT] residual width {hidden} -> {retained} using calibration PCA.")
    return model, {"slice_gpt.hidden": retained}
