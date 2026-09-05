"""Shared helpers for compression benchmark scripts.

Provides load_model_from_ckpt() and eval_perplexity() which replicate the
patterns from src/compress_model.py so that each method wrapper can stay concise.
"""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

# Allow importing project modules from src/
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.utils import DataReader, get_dataset
from models.utils import get_model
from optim.utils import eval as eval_model


# ---------------------------------------------------------------------------
# Config / checkpoint helpers
# ---------------------------------------------------------------------------

def load_config(ckpt_path: Path) -> SimpleNamespace:
    """Walk up from the checkpoint to find summary.json with training args."""
    for parent in ckpt_path.parents:
        candidate = parent / "summary.json"
        if candidate.exists():
            with open(candidate) as f:
                args_dict = json.load(f)["args"]
            return SimpleNamespace(**args_dict)
    raise FileNotFoundError(
        f"Could not find summary.json in any parent of {ckpt_path}. "
        "Place summary.json in the experiment directory."
    )


def load_model_from_ckpt(
    ckpt_path: Path,
    device: str = "cpu",
    datasets_dir: Path | None = None,
    tokenized_data_dir: Path | None = None,
):
    """Load model + config from a checkpoint produced by src/main.py.

    Returns:
        (model, cfg, val_reader)  — model is eval-mode on ``device``.
    """
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading config from summary.json ...")
    cfg = load_config(ckpt_path)
    cfg.use_pretrained = "none"
    if datasets_dir is not None:
        cfg.datasets_dir = str(datasets_dir)
    if tokenized_data_dir is not None:
        cfg.tokenized_data_dir = str(tokenized_data_dir)

    print(f"Building model ({cfg.model}) ...")
    model = get_model(cfg)

    print(f"Loading checkpoint weights ...")
    # Training checkpoints also contain optimizer state.  Loading the whole
    # payload directly on CUDA can consume several extra GiB until the local
    # checkpoint object is released.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device)
    model.eval()

    print("Building validation data reader ...")
    val_reader = DataReader(
        data_src=get_dataset(cfg)["val"],
        batch_size=cfg.batch_size,
        sequence_length=cfg.sequence_length,
        seed=cfg.data_seed,
        with_replacement=False,
        auto_shard=False,
    )

    return model, cfg, val_reader


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def eval_perplexity(model, val_reader, device: str = "cpu", eval_batches: int = 64):
    """Run eval and return (loss, perplexity)."""
    # DataReader.sample_batch() is stateful. Reset it so the dense baseline and
    # every compression method see exactly the same validation batches.
    val_reader.set_step(0)
    type_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if "cuda" in device else nullcontext()
    )
    model.eval()
    _, loss, ppl, _, _ = eval_model(
        model, val_reader, device=device,
        max_num_batches=eval_batches, ctx=type_ctx,
    )
    return loss, ppl


# ---------------------------------------------------------------------------
# DataLoader adapter
# ---------------------------------------------------------------------------

def make_calibration_dataloader(val_reader, n_batches: int = 16):
    """Wrap a DataReader so it can be iterated like a standard DataLoader.

    Each iteration yields the input token batch (x) only, shape (B, T).
    """
    val_reader.set_step(0)
    batches = []
    for _ in range(n_batches):
        x, _ = val_reader.sample_batch()
        batches.append(x)
    val_reader.set_step(0)
    return batches
