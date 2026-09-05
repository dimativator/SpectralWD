import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

from fineweb_edu import get_fineweb_edu_data


REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
PARQUET_HASHES = {
    "000_00000.parquet": "955ff462f3db09a52a4750e7a69901f89d64e48dfc936de0847c9f234f32b695",
    "000_00001.parquet": "776654388fdb2e5a1518b8d9d32543f418f1705d8bbaf408d221f08d76293680",
    "000_00002.parquet": "85b0bc930150d24b44041745b47ecb1433d961cb6974a31227430644008eec49",
    "000_00003.parquet": "34e77fe4ecc6075a2521d7716efb993636cd83ccefc2afc3fda3dbf35854c78b",
    "000_00004.parquet": "2f3f66cd93f0cf4919725e6dbcd1c240314a2866c9527870231e6148061ca73f",
}
TOKENIZED_HASHES = {
    "train.bin": "5a4c532cf1a142834f918f509fc678c178d8edc0df70eaffecc93a955b2f98af",
    "val.bin": "d3025934e78388200ace8ab058eef34a0dd5a0a3e961d08d637665f436c4996e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_hash: str) -> None:
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected_hash}, got {actual_hash}"
        )
    print(f"Verified {path}: {actual_hash}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    raw_dir = args.root / "sample" / "100BT"
    tokenized_dir = args.root / "tokenized"

    tokenized_paths = {
        name: tokenized_dir / name for name in TOKENIZED_HASHES
    }
    if all(path.exists() for path in tokenized_paths.values()):
        try:
            for name, path in tokenized_paths.items():
                verify(path, TOKENIZED_HASHES[name])
            print(
                f"H200-compatible FineWeb-Edu data already ready at {tokenized_dir}",
                flush=True,
            )
            return
        except RuntimeError as error:
            print(f"Existing tokenized data is invalid: {error}", flush=True)

    for path in tokenized_paths.values():
        if path.exists():
            print(f"Removing incomplete generated file {path}", flush=True)
            path.unlink()

    for name, expected_hash in PARQUET_HASHES.items():
        downloaded = hf_hub_download(
            repo_id="HuggingFaceFW/fineweb-edu",
            repo_type="dataset",
            revision=REVISION,
            filename=f"sample/100BT/{name}",
            local_dir=args.root,
        )
        verify(Path(downloaded), expected_hash)

    outputs = get_fineweb_edu_data(
        str(raw_dir),
        max_files=len(PARQUET_HASHES),
        tokenized_data_dir=str(tokenized_dir),
    )
    for split, path in outputs.items():
        verify(Path(path), TOKENIZED_HASHES[f"{split}.bin"])

    print(f"H200-compatible FineWeb-Edu data ready at {tokenized_dir}", flush=True)


if __name__ == "__main__":
    main()
