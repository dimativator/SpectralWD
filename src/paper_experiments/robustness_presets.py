"""Frozen post-step configurations for the arXiv corruption-seed replications."""

NOISE_LEVELS = (0.1, 0.25, 0.4, 0.6)
CORRUPTION_SEEDS = (101, 102, 103, 104, 105)

# Coefficients were selected before the five corruption-seed replications.
MNIST_COEFFICIENTS = {
    "mlp": {"l2": (0.5, 2.0, 3.0, 0.5), "spectral": (0.0, 5.0, 5.0, 5.0)},
    "gru": {"l2": (0.5, 0.1, 5.0, 5.0), "spectral": (10.0,) * 4},
}
BERT_PRESETS = {
    "ag_news_confirmatory": {
        "seed": 1, "split_seed": 4242, "exclude_split_seed": 1337,
        "l2": 500.0, "spectral": 30.0,
    },
    "dbpedia_14": {"seed": 2, "split_seed": 5242, "l2": 300.0, "spectral": 20.0},
    "yahoo_answers_topics": {
        "seed": 3, "split_seed": 6252, "l2": 300.0, "spectral": 20.0,
    },
    "yelp_review_full": {
        "seed": 4, "split_seed": 7252, "l2": 300.0, "spectral": 30.0,
    },
}


def arxiv_defaults(model: str, dataset: str, method: str, noise_frac: float) -> dict:
    if noise_frac not in NOISE_LEVELS:
        raise ValueError(f"arxiv requires noise-frac in {NOISE_LEVELS}")
    if model == "bert":
        if dataset not in BERT_PRESETS:
            raise ValueError("arxiv uses ag_news_confirmatory rather than ag_news")
        preset = BERT_PRESETS[dataset]
        coefficient = 0.0 if method == "no_wd" else preset[method]
        return {
            "seed": preset["seed"],
            "split_seed": preset["split_seed"],
            "exclude_split_seed": preset.get("exclude_split_seed", -1),
            "coefficient": coefficient,
        }
    coefficient = (
        0.0 if method == "no_wd"
        else MNIST_COEFFICIENTS[model][method][NOISE_LEVELS.index(noise_frac)]
    )
    return {"seed": 0, "coefficient": coefficient}
