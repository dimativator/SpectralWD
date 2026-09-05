import glob
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

tknzr = tiktoken.get_encoding("gpt2")


def get_fineweb_edu_data(
    datasets_dir,
    num_proc=5,
    max_files=None,
    tokenized_data_dir=None,
    require_tokenized=False,
):
    """To change the cache dir, run `export HF_HOME=/path/to/cache/` before running the code.

    Local parquet files are searched in this order:
      1. directly in datasets_dir (user points straight at the parquet folder)
      2. datasets_dir/fineweb-edu-100BT/sample/100BT/  (default layout)
    Tokenized .bin files are written next to the parquets that were found,
    or into datasets_dir/fineweb-edu-100BT/ when downloading from HuggingFace.
    """
    # Locate parquet files
    local_files = sorted(glob.glob(os.path.join(datasets_dir, "*.parquet")))
    if local_files:
        FWEB_DATA_PATH = datasets_dir
    else:
        fallback_dir = os.path.join(datasets_dir, "fineweb-edu-100BT", "sample", "100BT")
        local_files = sorted(glob.glob(os.path.join(fallback_dir, "*.parquet")))
        FWEB_DATA_PATH = os.path.join(datasets_dir, "fineweb-edu-100BT")

    if tokenized_data_dir is not None:
        FWEB_DATA_PATH = tokenized_data_dir

    data_paths = {
        split: os.path.join(FWEB_DATA_PATH, f"{split}.bin")
        for split in ("train", "val")
    }
    if require_tokenized:
        missing = [
            path
            for path in data_paths.values()
            if not os.path.isfile(path) or os.path.getsize(path) == 0
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise FileNotFoundError(
                "Pre-tokenized FineWeb-Edu data is required, but these files "
                f"are missing or empty: {missing_text}"
            )
        return data_paths

    if not os.path.exists(data_paths["train"]):
        os.makedirs(FWEB_DATA_PATH, exist_ok=True)

        if local_files:
            if max_files is not None:
                local_files = local_files[:max_files]
            print(f"Found {len(local_files)} local parquet files, loading from disk.")
            dataset = load_dataset(
                "parquet",
                data_files={"train": local_files},
                split="train",
                verification_mode="no_checks",
            )
        else:
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name="sample-100BT",
                split="train",
                streaming=False,
                verification_mode="no_checks",
            )

        split_dataset = dataset.train_test_split(
            test_size=0.0001, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(FWEB_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return data_paths


if __name__ == "__main__":
    get_fineweb_edu_data("./datasets/")
