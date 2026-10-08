# Spectral Weight Decay experiments

This directory is a standalone research-code snapshot for the paper experiments. It contains:

- fixed-horizon robustness experiments with an MLP, a row-sequential GRU, and BERT-base
- matched LLaMA pretraining with Adam and no weight decay, matrix L2 weight decay, or Spectral Weight Decay
- checkpoint compression benchmarks for truncated SVD, SliceGPT, ASVD, SVD-LLM, and Dobi-SVD

The training infrastructure in this release is based on [EPFL ML's `llm-baselines`](https://github.com/epfml/llm-baselines). The original license is included in [LICENSE](LICENSE).

Run every command from the repository root. The launchers print their fully resolved command or configuration with `--dry-run`, which should be checked before allocating a GPU.

## Setup

Python 3.10 or 3.11 is recommended. Create the environment with:

```bash
./setup.sh
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

`setup.sh` installs code dependencies only. It intentionally does not download FineWeb-Edu or model checkpoints.

## FineWeb-Edu dataset

LLaMA training reads FineWeb-Edu `sample/100BT` from local parquet shards. Download it once before starting training:

```bash
hf download HuggingFaceFW/fineweb-edu \
  --repo-type dataset \
  --include "sample/100BT/*" \
  --local-dir data/fineweb-edu
```

The parquet shards will be located at:

```text
data/fineweb-edu/sample/100BT/
```

The value passed to `--datasets-dir` must point directly to the directory containing the `*.parquet` files. The first LLaMA run tokenizes the selected shards and writes `train.bin` and `val.bin` to `--tokenized-data-dir`. Subsequent runs reuse those files.

The historical paper launcher uses the first five parquet shards by default. Keep the default for exact reproduction. Use `--finewebedu-max-files N` only for an explicitly documented data-scale ablation.

## LLaMA pretraining

The launcher provides the 124M, 257M, and 500M profiles used in the study. All profiles use seed 0, sequence length 1024, effective batch size 128, 2,000 warmup steps, and cosine decay.

Audit a command first:

```bash
PYTHONPATH=./src python -m paper_experiments.llama \
  --size 257m \
  --method spectral \
  --coefficient "$COEFFICIENT" \
  --datasets-dir data/fineweb-edu/sample/100BT \
  --tokenized-data-dir data/fineweb-edu-tokenized \
  --device cuda:0 \
  --dry-run
```

For custom LLaMA runs, set `COEFFICIENT` to the desired regularization strength. Run the three matched Adam variants:

```bash
# Adam, no weight decay
PYTHONPATH=./src python -m paper_experiments.llama \
  --size 257m --method no_wd \
  --datasets-dir data/fineweb-edu/sample/100BT \
  --tokenized-data-dir data/fineweb-edu-tokenized \
  --device cuda:0

# Adam with L2 weight decay
PYTHONPATH=./src python -m paper_experiments.llama \
  --size 257m --method l2 --coefficient "$COEFFICIENT" \
  --datasets-dir data/fineweb-edu/sample/100BT \
  --tokenized-data-dir data/fineweb-edu-tokenized \
  --device cuda:0

# Adam with Spectral Weight Decay
PYTHONPATH=./src python -m paper_experiments.llama \
  --size 257m --method spectral --coefficient "$COEFFICIENT" \
  --datasets-dir data/fineweb-edu/sample/100BT \
  --tokenized-data-dir data/fineweb-edu-tokenized \
  --device cuda:0
```

Replace `257m` with `124m` or `500m` to select another paper scale. Results are written under `exps/paper_llama/` unless `--results-base-folder` is supplied.

`no_wd` disables AdamW weight decay, and `l2` uses the requested L2 coefficient. `spectral` uses `AdamWSpectralL1Reg` on matrix weights. Non-matrix parameters retain the L2 decay configured by `--spectral-nonmatrix-weight-decay`.

## Weight-decay order

The robustness and LLaMA launchers accept `--wd-order pre|post`. Spectral WD defaults to `post`, and L2 WD defaults to `pre`. Pre-step decay uses the weights before the Adam update, while post-step decay uses the updated weights. Both are decoupled from Adam moments. For BERT, the same rule applies to `W - W0`. For example, append `--wd-order pre` to a spectral run or `--wd-order post` to an L2 run. Generated run names include the order.

Direct `src/main.py` runs use `--spectral_wd_order` and `--l2_wd_order`, with the same defaults. `--spectral_l1_reg_coupled` remains a separate gradient-regularization mode. The published robustness L2 runs used post-step decay, so reproduce them with `--wd-order post`.

## Robustness experiments

All robustness experiments use a fixed training horizon. The final epoch is the scientific checkpoint. Peak validation accuracy is diagnostic only, and test evaluation is performed once at the final horizon.

### arXiv configurations

Use `--preset arxiv` for the robustness coefficients and seeds used in the paper. It fixes model and split seeds, coefficients, architecture, and training horizon. `--noise-seed` selects one of the five label-corruption seeds, 101 through 105 (default: 101). Initialization, data splits, and training order remain fixed across these replications. Conflicting overrides of frozen parameters are rejected.

```bash
PYTHONPATH=./src python -m paper_experiments.robustness \
  --preset arxiv --model mlp --method spectral \
  --noise-frac 0.6 --noise-seed 101 \
  --datasets-dir data/mnist --device cuda:0 --dry-run

PYTHONPATH=./src python -m paper_experiments.robustness \
  --preset arxiv --model bert --dataset ag_news_confirmatory \
  --method spectral --noise-frac 0.6 --noise-seed 101 \
  --datasets-dir data/bert --device cuda:0 --dry-run
```

Remove `--dry-run` to train. Each replication gets a separate result directory. MNIST is downloaded by `torchvision`, and BERT datasets are downloaded by Hugging Face Datasets.

MNIST uses model/split seed 0, 3,000 training examples, 5,000 clean validation examples, and 60 epochs. The MLP has four hidden layers of width 2,048. The row-sequential GRU has two layers of hidden dimension 1,280. Spectral and L2 decay act on the same matrices.

BERT-base trains for 25 epochs. Embeddings and the bottom eight encoder blocks are frozen. Spectral WD and L2-SP regularize `W - W0` for the same matrices in the top four encoder blocks and pooler. The classifier, biases, and normalization parameters are excluded. Coefficients remain fixed across noise levels.

The confirmatory AG News split excludes the pilot training and validation examples selected with split seed 1337. Its test set contains held-out examples from the official training partition. The default BERT dataset for `--preset arxiv` is `ag_news_confirmatory`. `--data-seed` is an alias for `--noise-seed` on BERT.

Coefficients were calibrated using final clean-validation accuracy in separate tuning runs: independently at each noise level for MNIST, and at 60% noise for BERT. They were frozen before the five corruption-seed replications. The test results and clean validation were not used to select checkpoints.

To run all 360 post-step conditions:

```bash
for seed in 101 102 103 104 105; do
  for noise in 0.1 0.25 0.4 0.6; do
    for method in no_wd l2 spectral; do
      for model in mlp gru; do
        PYTHONPATH=./src python -m paper_experiments.robustness \
          --preset arxiv --model "$model" --method "$method" \
          --noise-frac "$noise" --noise-seed "$seed" --wd-order post \
          --datasets-dir data/mnist --device cuda:0
      done
      for dataset in ag_news_confirmatory dbpedia_14 yahoo_answers_topics yelp_review_full; do
        PYTHONPATH=./src python -m paper_experiments.robustness \
          --preset arxiv --model bert --dataset "$dataset" --method "$method" \
          --noise-frac "$noise" --noise-seed "$seed" --wd-order post \
          --datasets-dir data/bert --device cuda:0
      done
    done
  done
done
```

### Custom configurations

Set `COEFFICIENT` to the desired regularization strength for the example below.

Omit `--preset` to set coefficients and seeds manually. For MLP/GRU, `--seed` controls initialization, splits, and training order, while `--noise-seed` controls only corruption. Omitting `--noise-seed` preserves the original behavior, using `--seed` for corruption too. BERT retains its legacy defaults: model seed 0, split seed 1337, data seed 2000, and the official `ag_news` test set.

```bash
PYTHONPATH=./src python -m paper_experiments.robustness \
  --model gru --method spectral --coefficient "$COEFFICIENT" \
  --noise-frac 0.6 --seed 0 --noise-seed 102 \
  --datasets-dir data/mnist --device cuda:0
```

## Checkpoint compression

Each training run stores `summary.json` beside its checkpoint directory. Keep both when moving a checkpoint because the benchmark reconstructs the model from the saved training configuration.

Run all five paper methods with automatic rank selection:

```bash
PYTHONPATH=./src python -m paper_experiments.compression \
  --checkpoint exps/paper_llama/RUN/ckpts/latest/main.pt \
  --ranks auto \
  --datasets-dir data/fineweb-edu/sample/100BT \
  --tokenized-data-dir data/fineweb-edu-tokenized \
  --device cuda:0 \
  --output results/compression.md \
  --output-json results/compression.json
```

Repeat `--checkpoint PATH` to benchmark several dense checkpoints. Restrict the comparison with, for example, `--methods truncated_svd asvd svd_llm`. Explicit integer ranks may be supplied after `--ranks` instead of `auto`.

All methods target the same attention and MLP projections: `c_attn`, `c_proj`, `w1`, and `w2`. Embeddings and the output head remain dense. ASVD uses activation-aware scaling, SliceGPT performs calibrated residual-stream PCA and structural slicing, SVD-LLM uses whitening-aware factorization, and Dobi-SVD uses differentiable rank allocation with the non-remapped IPCA update. Validation and calibration readers are reset for every method so all rows use identical data. Add `--no-downstream` for a faster validation-loss-only run.

`rank=auto` is method-specific and does not impose an equal compression budget. Always report the measured parameter compression ratio when comparing methods.
