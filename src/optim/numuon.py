"""
nuMuon: Nuclear-Norm-Constrained Muon for Compressible LLM Training
Paper: https://arxiv.org/abs/2603.03597

nuMuon modifies Muon by constraining the update direction to a nuclear-norm ball,
encouraging learned weights to develop low-rank structure. Instead of orthogonalizing
all singular directions (Muon), nuMuon uses only the top-k, where k is controlled by
a rank fraction schedule that can anneal during training.
"""

import math
import os

import torch
import torch.distributed as dist

from optim.muon import zeropower_via_newtonschulz5

_NS_STEPS = 5  # default Newton-Schulz iterations (matches Muon default)


def _block_krylov_svd(G, k, L=2, oversample=8, B0=None):
    """
    Algorithm 3 from the nuMuon paper (Musco & Musco 2015).
    Builds a Krylov subspace with block size b=k+oversample, then extracts
    top-k singular triplets via a small SVD of the projected matrix.

    Args:
        G:  (m, n) float32 gradient matrix.
        k:  target rank.
        L:  number of Krylov iterations.
        oversample: extra vectors in the Krylov block (8 in the paper).
        B0: optional warm-start right block; from the previous step's V_k.

    Returns:
        U_k (m, k), S_k (k,), V_k (n, k)
    """
    _, n = G.shape
    b = min(n, k + oversample)

    # Run large CUDA matmuls in bfloat16; keep CPU tests and all factorizations
    # in float32. The final small SVD is always float32.
    matmul_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    Gbf = G.to(matmul_dtype)

    # --- Initialize B_0 ---
    if B0 is not None and B0.ndim == 2 and B0.shape[0] == n:
        B = B0[:, :b].float()
        if B.shape[1] < b:
            random = torch.randn(n, b - B.shape[1], device=G.device)
            B = torch.cat((B, random), dim=1)
    else:
        B = torch.randn(n, b, device=G.device)
    # CUDA QR is not implemented for bfloat16. Keep the bases in fp32 and
    # cast only for the large matrix multiplications.
    B, _ = torch.linalg.qr(B, mode="reduced")

    # --- Build block Krylov basis K = [B_1 | B_2 | ... | B_L] ---
    K_blocks = []
    for _ in range(L):
        T = Gbf @ B.to(matmul_dtype)    # (m, b)
        B = (Gbf.T @ T).float()         # (n, b)
        B, _ = torch.linalg.qr(B, mode="reduced")
        K_blocks.append(B)

    K = torch.cat(K_blocks, dim=1)   # (n, b*L)
    Q, _ = torch.linalg.qr(K, mode="reduced")

    # --- Project and compute small SVD in float32 for numerical stability ---
    T = G @ Q.float()                                           # (m, b*L)
    U_T, S_T, V_T = torch.linalg.svd(T, full_matrices=False)  # thin SVD

    U_k = U_T[:, :k]           # (m, k)
    S_k = S_T[:k]              # (k,)
    V_k = Q @ V_T.T[:, :k]    # (n, k)

    return U_k, S_k, V_k


def _numuon_svd_update(G, k, niter=2, oversample=8, B0=None):
    """
    Compute top-k singular direction update U_k @ V_k^T.
    - Full-rank: Newton-Schulz, which computes the same polar-factor update.
    - Low-rank: Block Krylov SVD (Algorithm 3 from the nuMuon paper),
      with optional warm-start B0 from the previous step's right singular vectors.

    Returns (update, V_k) where V_k can be stored for warm-starting next step.
    """
    m, n = G.shape
    full_rank = min(m, n)
    k = max(1, min(k, full_rank))

    if k == full_rank:
        update = zeropower_via_newtonschulz5(G.bfloat16(), steps=_NS_STEPS).to(G.dtype)
        return update, None

    # Low-rank path: Block Krylov SVD (float32 for numerical stability)
    orig_dtype = G.dtype
    U_k, _, V_k = _block_krylov_svd(
        G.float(), k, L=niter, oversample=oversample, B0=B0
    )
    update = (U_k @ V_k.T).to(orig_dtype)
    return update, V_k.to(orig_dtype).detach()


def _current_rank_fraction(
    r_start,
    r_end,
    schedule,
    step,
    total_steps,
    hold_fraction=0.1,
    decay_end_fraction=0.8,
):
    if r_end is None or schedule == "fixed" or total_steps is None or total_steps <= 0:
        return r_start
    t = min(step / total_steps, 1.0)
    if t <= hold_fraction:
        return r_start
    if t >= decay_end_fraction:
        return r_end
    t = (t - hold_fraction) / (decay_end_fraction - hold_fraction)
    if schedule == "cosine":
        return r_end + 0.5 * (r_start - r_end) * (1.0 + math.cos(math.pi * t))
    elif schedule == "linear":
        return r_start + (r_end - r_start) * t
    return r_start


class NuMuon(torch.optim.Optimizer):
    """
    nuMuon - Nuclear-Norm-Constrained Muon

    https://arxiv.org/abs/2603.03597

    Arguments:
        muon_params: Parameters to be optimized by nuMuon (>=2D, non-embedding).
        lr: Learning rate. (0.02 is a good default, same as Muon)
        momentum: SGD momentum coefficient. (0.95 is a good default)
        nesterov: Optional Nesterov-style momentum (disabled in the paper recipe).
        rank_fraction: Initial fraction of singular directions to use (0 < r <= 1).
        rank_fraction_final: Final rank fraction after schedule. None = fixed.
        rank_schedule: Annealing schedule for rank: 'fixed', 'cosine', or 'linear'.
        total_steps: Total training iterations (required for non-fixed schedule).
        svd_niter: Randomized block Krylov iterations (2 in the paper).
        svd_oversample: Extra Krylov block vectors (8 in the paper).
        adamw_params: Additional parameters to include in optimizer classification.
        adamw_lr: Learning rate for the internal AdamW.
        adamw_betas: Betas for the internal AdamW.
        adamw_eps: Epsilon for the internal AdamW.
        adamw_wd: Weight decay for the internal AdamW.
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=False,
        rank_fraction=1.0,
        rank_fraction_final=None,
        rank_schedule="cosine",
        total_steps=None,
        svd_niter=2,
        svd_oversample=8,
        rank_hold_fraction=0.1,
        rank_decay_end_fraction=0.8,
        weight_decay=0.1,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            rank_fraction=rank_fraction,
            rank_fraction_final=rank_fraction_final,
            rank_schedule=rank_schedule,
            total_steps=total_steps,
            svd_niter=svd_niter,
            svd_oversample=svd_oversample,
            rank_hold_fraction=rank_hold_fraction,
            rank_decay_end_fraction=rank_decay_end_fraction,
            weight_decay=weight_decay,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        muon_params = list(muon_params)
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        if not 0.0 <= rank_hold_fraction < rank_decay_end_fraction <= 1.0:
            raise ValueError(
                "Expected 0 <= rank_hold_fraction < "
                "rank_decay_end_fraction <= 1."
            )

        for p in muon_params:
            # Apply nuMuon to >=2D params that are not large embedding-like matrices
            self.state[p]["use_numuon"] = p.ndim >= 2 and p.size(0) < 10000
            self.state[p]["use_scalar_lion"] = p.ndim < 2
        for p in adamw_params:
            self.state[p]["use_numuon"] = False
            self.state[p]["use_scalar_lion"] = p.ndim < 2

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self):
        for group in self.param_groups:

            # ---- nuMuon update for matrix params ----
            params = [p for p in group["params"] if self.state[p]["use_numuon"]]
            lr = group["lr"]
            momentum = group["momentum"]
            svd_niter = group["svd_niter"]

            if "step" not in group:
                group["step"] = 0
            current_step = group["step"]

            r = _current_rank_fraction(
                group["rank_fraction"],
                group["rank_fraction_final"],
                group["rank_schedule"],
                current_step,
                group["total_steps"],
                group["rank_hold_fraction"],
                group["rank_decay_end_fraction"],
            )

            total_params = sum(p.numel() for p in params)
            # The flat buffer is only needed to exchange sharded updates. Avoid
            # allocating and copying an extra model-sized tensor on one GPU.
            updates_flat = None
            if self.world_size > 1:
                updates_flat = torch.zeros(
                    total_params,
                    device=params[0].device if params else group["params"][0].device,
                    dtype=torch.bfloat16,
                )
            curr_idx = 0

            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    g = p.grad
                    assert g is not None
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf.clone()

                    m, n = g.shape
                    k = max(1, math.ceil(r * min(m, n)))

                    # Top-k SVD update: U_k @ V_k^T (warm-start B0 from last step)
                    update, v_k = _numuon_svd_update(
                        g,
                        k,
                        niter=svd_niter,
                        oversample=group["svd_oversample"],
                        B0=state.get("krylov_V"),
                    )
                    if v_k is not None:
                        state["krylov_V"] = v_k  # warm-start for next step

                    # Muon's rectangular-matrix scaling. Do not normalize by k:
                    # the NuMuon LMO requires all k nonzero singular values to be 1.
                    scale = max(1.0, m / n) ** 0.5
                    update = update * scale

                    if updates_flat is None:
                        p.data.mul_(1 - lr * group["weight_decay"])
                        p.data.add_(update, alpha=-lr)
                    else:
                        updates_flat[curr_idx : curr_idx + p.numel()] = update.flatten()
                curr_idx += p.numel()

            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
                curr_idx = 0
                for p in params:
                    g = (
                        updates_flat[curr_idx : curr_idx + p.numel()]
                        .view_as(p.data)
                        .type_as(p.data)
                    )
                    p.data.mul_(1 - lr * group["weight_decay"])
                    p.data.add_(g, alpha=-lr)
                    curr_idx += p.numel()

            group["step"] += 1

            # ---- Lion for scalar (non-matrix) parameters ----
            params_lion = [
                p for p in group["params"] if self.state[p]["use_scalar_lion"]
            ]
            beta1, beta2 = group["adamw_betas"]
            for p in params_lion:
                grad = p.grad
                assert grad is not None
                state = self.state[p]
                if "lion_exp_avg" not in state:
                    state["lion_exp_avg"] = torch.zeros_like(grad)
                exp_avg = state["lion_exp_avg"]
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.add_(update.sign(), alpha=-lr)
                exp_avg.lerp_(grad, 1 - beta2)

            # ---- AdamW for embeddings and the language-model head ----
            params_adamw = [
                p
                for p in group["params"]
                if not self.state[p]["use_numuon"]
                and not self.state[p]["use_scalar_lion"]
            ]
            lr_adamw = group["adamw_lr_ratio"] * group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params_adamw:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "adamw_step" not in state:
                    state["adamw_step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["adamw_step"] += 1
                step = state["adamw_step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g_upd = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                bc_scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr_adamw * weight_decay)
                p.data.add_(g_upd, alpha=-lr_adamw / bc_scale)
