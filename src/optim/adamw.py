"""PyTorch AdamW with selectable decoupled weight-decay order."""

import torch


class AdamWWithOrder(torch.optim.AdamW):
    def __init__(self, params, l2_wd_order="pre", **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group.setdefault("l2_wd_order", l2_wd_order)
            if group["l2_wd_order"] not in ("pre", "post"):
                raise ValueError("l2_wd_order must be pre or post")
            if group["differentiable"] and group["l2_wd_order"] == "post":
                raise ValueError("post-step AdamW does not support differentiable=True")

    @torch.no_grad()
    def step(self, closure=None):
        post_groups = []
        for group in self.param_groups:
            if group.get("l2_wd_order", "pre") == "post":
                post_groups.append((group, group["weight_decay"]))
                group["weight_decay"] = 0.0
        try:
            loss = super().step(closure)
        finally:
            for group, decay in post_groups:
                group["weight_decay"] = decay
        # Fused AdamW can skip the whole update when GradScaler finds overflow.
        found_inf = getattr(self, "found_inf", None)
        if found_inf is not None and found_inf.item() != 0:
            return loss
        # Match AdamW's eligibility rule: parameters without gradients are skipped.
        for group, decay in post_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.mul_(1 - group["lr"] * decay)
        return loss
