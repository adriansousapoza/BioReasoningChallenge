# scripts/

Setup and utility scripts. These are run once (or occasionally) rather than as part of the main inference pipeline.

## Files

### `lumi_setup.sh`

One-time environment setup on LUMI: creates the Python venv, installs dependencies, downloads the Ollama binary, and pulls the `gpt-oss:120b` model weights.

### `setup_syncthing_lumi.sh`

Sets up Syncthing on the LUMI login node so experiment outputs sync automatically to your local machine over an SSH tunnel. Run from your local machine:

```bash
bash scripts/setup_syncthing_lumi.sh
```

This rsyncs the code to LUMI, installs Syncthing, and prints pairing instructions. See the script for full details.

### `warm_cache.py`

Pre-populates the mygene.info and STRING DB API caches for all genes in `data/train.csv` and `data/test.csv`. Run this before submitting a LUMI job to avoid rate-limited API calls during inference:

```bash
uv run python scripts/warm_cache.py
```

### `build_replogle_cache.py`

Downloads and processes the Replogle et al. 2022 K562 Perturb-seq dataset into `cache/replogle_k562_signatures.json`. Only needs to be run once (the output is committed to the repo). Re-run if the source data changes.

### `serve_with_logprobs_fix.py`

Drop-in replacement for `vllm serve` that patches a vLLM bug where `-inf` log-probabilities in JSON responses cause a serialization crash. Use this instead of `vllm serve` when you need logprobs:

```bash
uv run --extra serve python scripts/serve_with_logprobs_fix.py \
    openai/gpt-oss-120b --port 8000 --enforce-eager --no-enable-prefix-caching
```

For normal Track B inference (no logprobs needed), plain `vllm serve` works fine.
