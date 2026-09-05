#!/usr/bin/env python3
"""Compression benchmark: run all methods across a rank sweep and produce a Markdown table.

Usage:
    python ./src/compression/benchmark.py \\
        exps/run1/ckpts/latest/main.pt exps/run2/ckpts/latest/main.pt \\
        [--ranks 64 128 256 300 330] \\
        [--device cuda] \\
        [--eval_batches 64] \\
        [--calib_batches 16] \\
        [--output results.md]
"""

import argparse
import copy
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression._eval_utils import (
    load_model_from_ckpt,
    eval_perplexity,
    make_calibration_dataloader,
)

# ── downstream config ───────────────────────────────────────────────────────

_DOWNSTREAM_TASKS = [
    "arc_easy",
    "arc_challenge",
    "gsm8k_gold_bpb_5shot",
    "hellaswag",
    "piqa",
]
_TASK_METRIC = {
    "arc_easy":             "acc_v2",
    "arc_challenge":        "len_norm_v2",
    "gsm8k_gold_bpb_5shot": "bpb_v2",
    "hellaswag":            "len_norm_v2",
    "piqa":                 "len_norm_v2",
}
_TASK_HIGHER_BETTER = {
    "arc_easy":             True,
    "arc_challenge":        True,
    "gsm8k_gold_bpb_5shot": False,
    "hellaswag":            True,
    "piqa":                 True,
}
_TASK_SHORT = {
    "arc_easy":             "arc_e",
    "arc_challenge":        "arc_c",
    "gsm8k_gold_bpb_5shot": "gsm8k",
    "hellaswag":            "hella",
    "piqa":                 "piqa",
}
_METHOD_LABEL = {
    "baseline": "baseline (uncompressed)",
    "truncated_svd": "truncated SVD",
    "slice_gpt": "SliceGPT",
    "asvd": "ASVD",
    "svd_llm": "SVD-LLM",
    "dobi_svd": "Dobi-SVD",
}


def _hf_tokenizer_name(cfg) -> str:
    """Resolve HF tokenizer name from cfg.tokenizer, matching data/utils.get_tokenizer."""
    tok = getattr(cfg, "tokenizer", "gpt2")
    return "mistralai/Mistral-7B-v0.1" if tok == "mistral" else "gpt2"


def _run_downstream(model, cfg, device: str) -> dict:
    """Return {task_name: score}. Empty dict if olmo_eval not installed."""
    try:
        from olmo_eval import HFTokenizer, ICLMetric, build_task
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer
    except ImportError:
        print("  [downstream] olmo_eval / transformers not installed – skipping.")
        return {}

    tokenizer_name = _hf_tokenizer_name(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_tokenizer = HFTokenizer(
        tokenizer_name,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    type_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if "cuda" in device else nullcontext()
    )
    seq_len   = getattr(cfg, "sequence_length", 1024)
    batch_sz  = getattr(cfg, "eval_batch_size", 8)

    def _to_dev(v):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, dict):
            return {k: _to_dev(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(_to_dev(x) for x in v)
        return v

    scores = {}
    model.eval()

    for task_name in _DOWNSTREAM_TASKS:
        try:
            task_dataset = build_task(task_name, eval_tokenizer, model_ctx_len=seq_len)
        except Exception as exc:
            print(f"  [downstream] {task_name}: {exc}")
            scores[task_name] = float("nan")
            continue

        metric = ICLMetric(metric_type=task_dataset.metric_type)
        if hasattr(metric, "sync_on_compute"):
            metric.sync_on_compute = False
        if hasattr(metric, "_to_sync"):
            metric._to_sync = False
        if hasattr(metric, "to"):
            metric = metric.to(device)

        dl = DataLoader(
            task_dataset,
            batch_size=batch_sz,
            collate_fn=task_dataset.collate_fn,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )

        with torch.no_grad():
            for batch in dl:
                batch     = _to_dev(batch)
                input_ids = batch["input_ids"]
                targets   = input_ids.clone()
                if "label_mask" in batch:
                    targets.masked_fill_(~batch["label_mask"], -1)
                if "attention_mask" in batch:
                    targets.masked_fill_(batch["attention_mask"] == 0, -1)
                targets = torch.nn.functional.pad(targets[..., 1:], (0, 1), value=-1)
                with type_ctx:
                    outputs = model(input_ids, targets=targets, get_logits=True)
                metric.update(batch, outputs["logits"])

        raw = metric.compute()
        if isinstance(raw, dict):
            v2 = {k: v for k, v in raw.items() if str(k).endswith("_v2")}
            raw = v2 if v2 else raw

        key = _TASK_METRIC[task_name]
        if isinstance(raw, dict):
            val = raw.get(key, next(iter(raw.values())))
        else:
            val = raw
        scores[task_name] = float(val.detach().cpu()) if isinstance(val, torch.Tensor) else float(val)

    return scores


# ── compression wrappers ────────────────────────────────────────────────────
# Methods supporting per-layer effective-rank compression via rank="auto".
# Covers attention + MLP projections so target-module methods (svd_llm, asvd,
# fwsvd, ...) compress the same set of Linears as truncated_svd, which factors
# every Linear except the embeddings / lm_head. Names span this repo's llama
# (c_attn, c_proj, w1, w2) and HF-style (q/k/v/o/gate/up/down_proj) layouts.
_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "c_attn", "c_proj", "c_fc",
    "w1", "w2",
)
_AUTO_METHODS = ("truncated_svd", "svd_llm", "asvd", "slice_gpt", "dobi_svd")

# Set True once --target_modules is supplied, so truncated_svd restricts to the
# same layer subset the other (target-module) methods already use.
_TARGET_MODULES_OVERRIDDEN = False


def _lowrank_comp_info(model) -> dict:
    """Per-layer retained rank for a model whose Linears were factored to LowRankLinear."""
    from models.compress import LowRankLinear
    info = {}
    for name, m in model.named_modules():
        if isinstance(m, LowRankLinear):
            info[name] = m.B.out_features  # B: (in -> rank)
    return info


def _target_comp_info(model, rank_fn) -> dict:
    """Per-layer retained rank for in-place methods that approximate target Linears.

    ``rank_fn(out_features, in_features) -> retained_rank``; layers that are
    not actually reduced (retained >= min(out, in)) are omitted.
    """
    info = {}
    for name, m in model.named_modules():
        if not any(name.endswith(t) for t in _TARGET_MODULES):
            continue
        if not isinstance(m, torch.nn.Linear):
            continue
        out_f, in_f = m.weight.shape
        r = max(1, min(int(rank_fn(out_f, in_f)), out_f, in_f))
        if r < min(out_f, in_f):
            info[name] = r
    return info


def _auto_comp_info(model) -> dict:
    """Per-layer effective rank for in-place methods run with rank="auto".

    Must be called on the *uncompressed* weights (before the method overwrites
    them), so the reported ranks match what the method's own "auto" branch
    computes from ``effective_rank(W)``.
    """
    from models.compress import effective_rank
    info = {}
    for name, m in model.named_modules():
        if not any(name.endswith(t) for t in _TARGET_MODULES):
            continue
        if not isinstance(m, torch.nn.Linear):
            continue
        out_f, in_f = m.weight.shape
        r = max(1, min(round(effective_rank(m.weight.data)), out_f, in_f))
        if r < min(out_f, in_f):
            info[name] = r
    return info


def _apply_truncated_svd(model, rank, cfg, calib_data, device):
    from models.compress import (
        compress_model_svd,
        compress_model_svd_adaptive,
        compress_embeddings_inplace,
    )
    only = _TARGET_MODULES if _TARGET_MODULES_OVERRIDDEN else None
    if rank == "auto":
        compress_model_svd_adaptive(model, skip_names=("lm_head", "wte", "wpe"), only_names=only)
    else:
        compress_model_svd(model, rank=rank, skip_names=("lm_head", "wte", "wpe"), only_names=only)
    emb_info = compress_embeddings_inplace(model, rank, _TARGET_MODULES)
    model.eval()
    return {**_lowrank_comp_info(model), **emb_info}


def _apply_asvd(model, rank, cfg, calib_data, device):
    from compression.asvd import collect_activation_stats, apply_asvd
    stats = collect_activation_stats(
        model, calib_data, n_batches=len(calib_data), device=device,
    )
    from models.compress import compress_embeddings_inplace
    _, comp_info = apply_asvd(model, rank=rank, activation_stats=stats, target_modules=_TARGET_MODULES)
    comp_info.update(compress_embeddings_inplace(model, rank, _TARGET_MODULES))
    model.eval()
    return comp_info


def _apply_fwsvd(model, rank, cfg, calib_data, device):
    from compression.fwsvd import collect_fisher_stats, apply_fwsvd
    stats = collect_fisher_stats(
        model, calib_data, n_batches=len(calib_data), device=device,
    )
    apply_fwsvd(model, rank=rank, fisher_stats=stats, target_modules=_TARGET_MODULES)
    model.eval()
    return _target_comp_info(model, lambda o, i: rank)


def _apply_slice_gpt(model, rank, cfg, calib_data, device):
    from compression.slice_gpt import apply_slice_gpt
    _, comp_info = apply_slice_gpt(
        model,
        dataloader=calib_data,
        n_batches_pca=len(calib_data),
        target_modules=_TARGET_MODULES,
        device=device,
        rank=rank,
    )
    model.eval()
    return comp_info


def _apply_svd_llm(model, rank, cfg, calib_data, device):
    from compression.svd_llm import collect_input_covariance, apply_svd_llm
    stats = collect_input_covariance(
        model, calib_data, n_batches=len(calib_data), device=device,
    )
    from models.compress import compress_embeddings_inplace
    _, comp_info = apply_svd_llm(
        model, rank=rank, whitening_stats=stats, target_modules=_TARGET_MODULES,
    )
    comp_info.update(compress_embeddings_inplace(model, rank, _TARGET_MODULES))
    model.eval()
    return comp_info


def _apply_dobi_svd(model, rank, cfg, calib_data, device):
    from compression.dobi_svd import collect_output_covariances, apply_dobi_svd
    stats = collect_output_covariances(
        model,
        calib_data,
        target_modules=_TARGET_MODULES,
        n_batches=len(calib_data),
        device=device,
    )
    _, comp_info = apply_dobi_svd(
        model,
        rank=rank,
        output_covariances=stats,
        target_modules=_TARGET_MODULES,
        device=device,
    )
    model.eval()
    return comp_info


# (apply_fn, needs_calibration_data)
METHODS = {
    "truncated_svd": (_apply_truncated_svd, False),
    "asvd":          (_apply_asvd,          True),
    "fwsvd":         (_apply_fwsvd,         True),
    "slice_gpt":     (_apply_slice_gpt,     True),
    "svd_llm":       (_apply_svd_llm,       True),
    "dobi_svd":      (_apply_dobi_svd,      True),
}


def _compression_rate(orig_model, compressed_model, comp_info: dict | None = None) -> float:
    """Ratio of original to compressed parameters.

    Benchmark callers pass the instantiated compressed model so the count is
    exact.  The legacy two-argument form accepts ``comp_info`` and remains
    useful for callers that only have a rank map.
    """
    total = sum(p.numel() for p in orig_model.parameters())
    if comp_info is None:
        comp_info = compressed_model
        named = dict(orig_model.named_modules())
        comp_total = total
        for name, rank in comp_info.items():
            module = named.get(name)
            if module is None or not hasattr(module, "weight"):
                continue
            out_features, in_features = module.weight.shape
            comp_total += rank * (out_features + in_features) - out_features * in_features
        return total / comp_total if comp_total > 0 else float("inf")
    comp_total = sum(p.numel() for p in compressed_model.parameters())
    return total / comp_total if comp_total > 0 else float("inf")


# ── markdown table ──────────────────────────────────────────────────────────

def _fmt_loss_delta(delta: float) -> str:
    s = f"{delta:+.4f}"
    return f"**{s}**" if delta > 0.01 else s


def _fmt_metric_delta(delta: float, higher_better: bool) -> str:
    s = f"{delta:+.4f}"
    bad = (delta < -0.005) if higher_better else (delta > 0.005)
    return f"**{s}**" if bad else s


def _build_table(rows: list, tasks: list) -> str:
    short = [_TASK_SHORT[t] for t in tasks]

    header = ["Method", "Rank", "comp_rate", "val_loss", "Δval_loss"]
    for s in short:
        header += [s, f"Δ{s}"]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]

    for r in rows:
        is_base = r["method"] == "baseline"
        cells = [
            _METHOD_LABEL.get(r["method"], r["method"]),
            str(r["rank"]),
            "-" if is_base else f"{r['comp_rate']:.2f}×",
            f"{r['val_loss']:.4f}",
            "-" if is_base else _fmt_loss_delta(r["val_loss"] - r["base_val_loss"]),
        ]
        for t in tasks:
            score = r.get(t, math.nan)
            base  = r.get(f"base_{t}", math.nan)
            cells.append(f"{score:.4f}" if not math.isnan(score) else "N/A")
            if is_base or math.isnan(score) or math.isnan(base):
                cells.append("-")
            else:
                cells.append(_fmt_metric_delta(score - base, _TASK_HIGHER_BETTER[t]))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ── entry point ─────────────────────────────────────────────────────────────

def main():
    global _DOWNSTREAM_TASKS
    parser = argparse.ArgumentParser(description="Compression benchmark.")
    parser.add_argument("ckpt_paths", nargs="+", type=Path)
    parser.add_argument("--names", nargs="*", default=[],
                        help="Display names for each checkpoint (in the same order). "
                             "Falls back to experiment_name from cfg if not provided.")
    parser.add_argument("--methods", nargs="+", default=None, choices=list(METHODS),
                        help="Compression methods to run (in the given order). "
                             "Defaults to all: " + ", ".join(METHODS) + ".")
    def _rank_type(v):
        return v if v == "auto" else int(v)
    parser.add_argument("--ranks", nargs="+", type=_rank_type, default=[64, 128, 256, 300, 330],
                        help='Ranks to sweep. Use "auto" to compress each layer to its '
                             'effective rank (supported by: ' + ", ".join(_AUTO_METHODS) + ").")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--datasets_dir",
        type=Path,
        default=None,
        help="Override datasets_dir stored in each checkpoint summary.json.",
    )
    parser.add_argument(
        "--tokenized_data_dir",
        type=Path,
        default=None,
        help="Override tokenized_data_dir stored in each checkpoint summary.json.",
    )
    parser.add_argument("--eval_batches", default=64, type=int)
    parser.add_argument("--calib_batches", default=16, type=int)
    parser.add_argument("--no_downstream", action="store_true",
                        help="Skip downstream eval (faster, val_loss only).")
    parser.add_argument("--downstream_tasks", nargs="+", choices=list(_DOWNSTREAM_TASKS), default=None,
                        help="Evaluate only this subset of downstream tasks.")
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--output_json", default=None, type=Path,
                        help="Write machine-readable rows after every completed method.")
    parser.add_argument("--target_modules", nargs="+", default=None,
                        help="Restrict compression to layers whose name ends with one "
                             "of these suffixes (e.g. q_proj c_attn w1). Defaults to "
                             "all attention + MLP projections.")
    args = parser.parse_args()

    if args.target_modules:
        global _TARGET_MODULES, _TARGET_MODULES_OVERRIDDEN
        _TARGET_MODULES = tuple(args.target_modules)
        _TARGET_MODULES_OVERRIDDEN = True
        print(f"Compressing only layers: {', '.join(_TARGET_MODULES)}")

    selected_methods = args.methods if args.methods else list(METHODS)
    if args.downstream_tasks:
        _DOWNSTREAM_TASKS = list(args.downstream_tasks)

    all_sections: list[str] = []

    for idx, ckpt_path in enumerate(args.ckpt_paths):
        print(f"\n{'='*60}\nCheckpoint: {ckpt_path}\n{'='*60}")

        model, cfg, val_reader = load_model_from_ckpt(
            ckpt_path,
            device=args.device,
            datasets_dir=args.datasets_dir,
            tokenized_data_dir=args.tokenized_data_dir,
        )
        # deepcopy the whole model, not just state_dict — some compression methods
        # (e.g. truncated_svd) replace nn.Linear with factored A/B modules, so the
        # architecture changes and state_dict restore would fail.
        orig_model = copy.deepcopy(model)

        run_ds = not args.no_downstream

        # baseline
        print("\nBaseline ...")
        base_loss, base_ppl = eval_perplexity(model, val_reader, args.device, args.eval_batches)
        print(f"  val_loss={base_loss:.4f}  ppl={base_ppl:.2f}")

        base_ds: dict[str, float] = {}
        if run_ds:
            print("  Downstream eval ...")
            base_ds = _run_downstream(model, cfg, args.device)
            for t, v in base_ds.items():
                print(f"    {_TASK_SHORT[t]}={v:.4f}")

        tasks = [t for t in _DOWNSTREAM_TASKS if t in base_ds]

        baseline_row: dict = {
            "method": "baseline",
            "rank": "-",
            "val_loss": base_loss,
            "base_val_loss": base_loss,
            **base_ds,
            **{f"base_{t}": v for t, v in base_ds.items()},
        }
        rows = [baseline_row]
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(rows, indent=2, allow_nan=True))

        # sweep
        for method_name in selected_methods:
            apply_fn, needs_calib = METHODS[method_name]
            for rank in args.ranks:
                if rank == "auto" and method_name not in _AUTO_METHODS:
                    print(f"\n[{method_name}  rank=auto]  skipped (auto unsupported)")
                    continue
                print(f"\n[{method_name}  rank={rank}]")
                model = copy.deepcopy(orig_model)
                model.eval()

                calib_data = None
                if needs_calib:
                    calib_data = make_calibration_dataloader(val_reader, args.calib_batches)

                try:
                    comp_info = apply_fn(model, rank, cfg, calib_data, args.device)
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    continue

                comp_rate = _compression_rate(orig_model, model, comp_info or {})

                comp_loss, comp_ppl = eval_perplexity(
                    model, val_reader, args.device, args.eval_batches,
                )
                print(
                    f"  comp_rate={comp_rate:.2f}x  val_loss={comp_loss:.4f}  ppl={comp_ppl:.2f}"
                    f"  Δval_loss={comp_loss - base_loss:+.4f}"
                )

                comp_ds: dict[str, float] = {}
                if run_ds:
                    comp_ds = _run_downstream(model, cfg, args.device)
                    for t, v in comp_ds.items():
                        print(f"    {_TASK_SHORT[t]}={v:.4f}  Δ={v - base_ds.get(t, 0):+.4f}")

                rows.append({
                    "method": method_name,
                    "rank": rank,
                    "comp_rate": comp_rate,
                    "val_loss": comp_loss,
                    "base_val_loss": base_loss,
                    **comp_ds,
                    **{f"base_{t}": base_ds.get(t, math.nan) for t in comp_ds},
                })
                if args.output_json:
                    args.output_json.parent.mkdir(parents=True, exist_ok=True)
                    args.output_json.write_text(json.dumps(rows, indent=2, allow_nan=True))

        exp_name = (
            args.names[idx]
            if args.names and idx < len(args.names)
            else getattr(cfg, "experiment_name", ckpt_path.as_posix())
        )
        table    = _build_table(rows, tasks)
        section  = f"## {exp_name}\n\n{table}\n"
        all_sections.append(section)
        print(f"\n{section}")

    md = "\n".join(all_sections)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"Table written to {args.output}")


if __name__ == "__main__":
    main()
