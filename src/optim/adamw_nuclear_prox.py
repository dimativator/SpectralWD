"""AdamW with *proximal* nuclear-norm weight decay (singular-value soft-thresholding).

This is the faithful proximal-gradient version of nuclear-norm regularization,
in contrast to ``adamw_spectral_L1_reg.py`` which takes a *subgradient* step
(subtracting U V^T) and therefore only shifts the whole spectrum down without
ever zeroing singular values.

For the regularizer R(W) = coef * ||W||_* the proximal operator is
singular-value soft-thresholding (SVT, Cai-Candes-Shen 2010):

    prox_{tau R}(Z) = U * diag(max(sigma_i - tau, 0)) * V^T,   tau = lr * coef

Singular values below the threshold are set to *exactly zero*, producing genuine
low rank instead of a uniformly shrunk spectrum.

The SVD used for the threshold step is configurable via ``svd_mode``:

    "full"                - exact torch.linalg.svd over all singular values.
    "randomized"          - randomized SVD (Halko et al. 2011) approximating the
                            full decomposition; faster on large matrices. The
                            sketch size is min(max_rank, min(m, n)); with
                            max_rank=None it defaults to min(m, n).
    "truncated-randomized"- randomized SVD with a *hard* rank cap at max_rank
                            (required), then soft-thresholding within that cap.
                            Cheapest and most aggressive: the tail beyond
                            max_rank is dropped outright.

To amortize the SVD cost the prox can be applied every ``prox_interval`` steps;
the threshold is then accumulated as tau = lr * coef * prox_interval so the
average shrinkage per step is unchanged.
"""

import math

import torch


def _randomized_svd(A, q, oversample=10, n_iter=2):
    """Randomized SVD (Halko, Martinsson, Tropp 2011) returning the top-q triplets.

    Args:
        A: (m, n) matrix (float32 recommended).
        q: target number of singular triplets.
        oversample: extra random projections for accuracy.
        n_iter: power iterations (improves accuracy on slowly decaying spectra).

    Returns:
        U (m, q), S (q,), Vh (q, n).
    """
    m, n = A.shape
    full_rank = min(m, n)
    q = max(1, min(q, full_rank))
    r = min(q + oversample, full_rank)

    # Sketch the range of A.
    G = torch.randn(n, r, device=A.device, dtype=A.dtype)
    Y = A @ G  # (m, r)
    Q, _ = torch.linalg.qr(Y)  # (m, r)

    # Power iterations to sharpen the captured subspace, re-orthonormalizing
    # each half-step for numerical stability.
    for _ in range(n_iter):
        Z, _ = torch.linalg.qr(A.T @ Q)  # (n, r)
        Q, _ = torch.linalg.qr(A @ Z)  # (m, r)

    # Small SVD on the projected matrix.
    B = Q.T @ A  # (r, n)
    U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_b  # (m, r)

    return U[:, :q], S[:q], Vh[:q, :]


def _singular_value_threshold(W, tau, svd_mode, max_rank, oversample, n_iter):
    """Return the SVT of W: U * diag(max(sigma - tau, 0)) * V^T.

    Returns the reconstructed matrix (same dtype as W) keeping only the
    singular values that survive the threshold.
    """
    orig_dtype = W.dtype
    A = W.float()
    m, n = A.shape

    if svd_mode == "full":
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    elif svd_mode == "randomized":
        q = min(max_rank, min(m, n)) if max_rank is not None else min(m, n)
        U, S, Vh = _randomized_svd(A, q, oversample=oversample, n_iter=n_iter)
    elif svd_mode == "truncated-randomized":
        if max_rank is None:
            raise ValueError(
                "svd_mode='truncated-randomized' requires nuclear_prox_max_rank to be set."
            )
        U, S, Vh = _randomized_svd(A, max_rank, oversample=oversample, n_iter=n_iter)
    else:
        raise ValueError(f"Unknown svd_mode: {svd_mode}")

    # Soft-threshold the singular values.
    S = torch.clamp(S - tau, min=0.0)

    # Keep only the surviving (non-zero) components for a cheap reconstruction.
    keep = S > 0
    if not torch.any(keep):
        return torch.zeros_like(W)
    U = U[:, keep]
    S = S[keep]
    Vh = Vh[keep, :]

    recon = (U * S.unsqueeze(0)) @ Vh
    return recon.to(orig_dtype)


class AdamWNuclearProx(torch.optim.Optimizer):
    """AdamW with proximal (singular-value soft-thresholding) nuclear-norm decay.

    The AdamW core matches the standard PyTorch/llm-baselines implementation. After
    the adaptive update, 2-D parameters receive the nuclear-norm prox step (applied
    to the gradient-stepped weights, faithful to proximal gradient descent). Non-2-D
    parameters get ordinary decoupled weight decay.

    Arguments:
        spectral_l1_reg_coef: the nuclear-norm coefficient (threshold = lr * coef).
            Named to match the scheduling hooks already in optim/base.py.
        prox_interval: apply SVT every this many steps (threshold accumulated).
        svd_mode: "full" | "randomized" | "truncated-randomized".
        max_rank: rank cap / sketch size for the randomized modes.
        svd_oversample: oversampling for randomized SVD.
        svd_power_iters: power iterations for randomized SVD.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        spectral_l1_reg_coef=0.1,
        prox_interval=1,
        svd_mode="full",
        max_rank=None,
        svd_oversample=10,
        svd_power_iters=2,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if prox_interval < 1:
            raise ValueError("prox_interval must be >= 1")
        if svd_mode not in ("full", "randomized", "truncated-randomized"):
            raise ValueError("Invalid svd_mode: {}".format(svd_mode))
        if svd_mode == "truncated-randomized" and max_rank is None:
            raise ValueError("svd_mode='truncated-randomized' requires max_rank to be set.")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            spectral_l1_reg_coef=spectral_l1_reg_coef,
            prox_interval=prox_interval,
            svd_mode=svd_mode,
            max_rank=max_rank,
            svd_oversample=svd_oversample,
            svd_power_iters=svd_power_iters,
        )

        super(AdamWNuclearProx, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWNuclearProx, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            coef = group["spectral_l1_reg_coef"]
            prox_interval = group["prox_interval"]
            svd_mode = group["svd_mode"]
            max_rank = group["max_rank"]
            oversample = group["svd_oversample"]
            n_iter = group["svd_power_iters"]

            for p in group["params"]:
                grad = p.grad

                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported!")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["first_momentum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["second_momentum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m = state["first_momentum"]
                v = state["second_momentum"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                is_matrix = p.data.ndim == 2

                # Ordinary decoupled weight decay for non-matrix params.
                if not is_matrix:
                    p.data.mul_(1 - lr * wd)

                # Adaptive (AdamW) update -- this is the gradient step. The
                # nuclear-norm prox is applied afterwards to the stepped weights.
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.data.addcdiv_(m, denom, value=-(lr / bias_correction1))

                # Proximal nuclear-norm step (SVT) on 2-D weights, every
                # prox_interval steps with an accumulated threshold so the
                # average per-step shrinkage stays at lr * coef.
                if is_matrix and coef > 0 and state["step"] % prox_interval == 0:
                    tau = lr * coef * prox_interval
                    p.data.copy_(
                        _singular_value_threshold(
                            p.data, tau, svd_mode, max_rank, oversample, n_iter
                        )
                    )

        return loss
