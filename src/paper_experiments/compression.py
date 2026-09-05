"""Benchmark paper compression methods on one or more dense checkpoints."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PAPER_METHODS = (
    "truncated_svd",
    "slice_gpt",
    "asvd",
    "svd_llm",
    "dobi_svd",
)


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "./src/compression/benchmark.py",
        *(str(path) for path in args.checkpoint),
        "--methods",
        *args.methods,
        "--ranks",
        *(str(rank) for rank in args.ranks),
        "--device",
        args.device,
        "--datasets_dir",
        str(args.datasets_dir),
        "--tokenized_data_dir",
        str(args.tokenized_data_dir),
        "--eval_batches",
        str(args.eval_batches),
        "--calib_batches",
        str(args.calib_batches),
        "--target_modules",
        "c_attn",
        "c_proj",
        "w1",
        "w2",
    ]
    if args.output is not None:
        command.extend(["--output", str(args.output)])
    if args.output_json is not None:
        command.extend(["--output_json", str(args.output_json)])
    if args.no_downstream:
        command.append("--no_downstream")
    return command


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=PAPER_METHODS, default=list(PAPER_METHODS)
    )
    parser.add_argument("--ranks", nargs="+", default=["auto"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("data/fineweb-edu/sample/100BT"),
    )
    parser.add_argument(
        "--tokenized-data-dir", type=Path, default=Path("data/fineweb-edu-tokenized")
    )
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--no-downstream", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    command = build_command(args)
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
