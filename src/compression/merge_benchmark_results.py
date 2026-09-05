#!/usr/bin/env python3
"""Merge sharded compression benchmark JSON files into one compact table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LABELS = {
    "baseline": "baseline (uncompressed)",
    "truncated_svd": "truncated SVD",
    "slice_gpt": "SliceGPT",
    "asvd": "ASVD",
    "svd_llm": "SVD-LLM",
    "dobi_svd": "Dobi-SVD",
}
ORDER = tuple(LABELS)
TASKS = (("arc_easy", "ARC-E"), ("hellaswag", "HellaSwag"), ("piqa", "PIQA"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = {}
    for path in args.inputs:
        for row in json.loads(path.read_text()):
            rows[(row["method"], str(row["rank"]))] = row
    ordered = []
    for method in ORDER:
        matches = [row for (name, _rank), row in rows.items() if name == method]
        ordered.extend(matches)
    baseline = next(row for row in ordered if row["method"] == "baseline")
    header = ["Method", "Compression", "val loss", "Δval loss", *(label for _, label in TASKS)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in ordered:
        is_baseline = row["method"] == "baseline"
        cells = [
            LABELS.get(row["method"], row["method"]),
            "–" if is_baseline else f"{row['comp_rate']:.2f}×",
            f"{row['val_loss']:.4f}",
            "–" if is_baseline else f"{row['val_loss'] - baseline['val_loss']:+.4f}",
        ]
        for task, _label in TASKS:
            score = row.get(task, math.nan)
            cells.append("N/A" if math.isnan(score) else f"{score:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
