#!/usr/bin/env bash
# lumi_setup.sh — One-shot setup for LUMI deployment.
# Run from your LOCAL machine (has internet + SSH access to LUMI).
#
# Usage:
#   bash scripts/lumi_setup.sh
#
# Prereqs:
#   - SSH key configured: ssh lumi.csc.fi works
#   - GRN data in ../GRN\ Inference/GRN-Embeddings/data/
#   - data/train.csv and data/test.csv present locally

set -euo pipefail

# ── Config (edit these) ───────────────────────────────────────────────────
LUMI_USER="sousapoz"
LUMI_HOST="lumi.csc.fi"
SSH_KEY="$HOME/.ssh/id_rsa_lumi"
PROJECT="project_465002610"
SCRATCH="/scratch/$PROJECT/$LUMI_USER"
OLLAMA_VERSION="v0.30.7"

# Local paths
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GRN="$REPO/../GRN Inference/GRN-Embeddings/data"
CHALLENGE="$REPO/.."

SSH="ssh -i $SSH_KEY $LUMI_USER@$LUMI_HOST"
RSYNC="rsync -av -e \"ssh -i $SSH_KEY\""

echo "========================================"
echo "LUMI Setup: $LUMI_USER@$LUMI_HOST"
echo "Scratch:    $SCRATCH"
echo "========================================"

# ── Step 1: Create directories ────────────────────────────────────────────
echo
echo ">>> Step 1/6: Creating scratch directories..."
$SSH "mkdir -p $SCRATCH/{code,data/grn,model/ollama,cache,outputs/track_b,scripts,bin/bin}"

# ── Step 2: Transfer code + data ─────────────────────────────────────────
echo
echo ">>> Step 2/6: Transferring code and data..."
rsync -av --exclude='outputs/' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.git/' \
    -e "ssh -i $SSH_KEY" \
    "$REPO/" "$LUMI_USER@$LUMI_HOST:$SCRATCH/code/"

rsync -av -e "ssh -i $SSH_KEY" \
    "$CHALLENGE/train.csv" \
    "$CHALLENGE/test.csv" \
    "$CHALLENGE/replogle_k562_signatures.json" \
    "$LUMI_USER@$LUMI_HOST:$SCRATCH/data/"

echo "Transferring GRN data (~200 MB)..."
rsync -av -e "ssh -i $SSH_KEY" \
    "$GRN/DoRothEA/dorothea_mm.csv" \
    "$GRN/TFLink/TFLink_Mus_musculus_interactions_All_simpleFormat_v1.0.tsv.gz" \
    "$GRN/GenePT/GenePT_emebdding_v2/NCBI_UniProt_summary_of_genes.json" \
    "$LUMI_USER@$LUMI_HOST:$SCRATCH/data/grn/"

# ── Step 3: Install Ollama (ROCm) ─────────────────────────────────────────
echo
echo ">>> Step 3/6: Installing Ollama $OLLAMA_VERSION (ROCm build)..."
$SSH "
  if [ -f $SCRATCH/bin/bin/ollama ]; then
    echo 'Ollama already installed: '
    $SCRATCH/bin/bin/ollama --version
  else
    # Download base binary
    curl -fsSL https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst \
        -o $SCRATCH/ollama_base.tar.zst
    zstd -d $SCRATCH/ollama_base.tar.zst -o $SCRATCH/ollama_base.tar
    tar -xf $SCRATCH/ollama_base.tar -C $SCRATCH/bin/
    rm $SCRATCH/ollama_base.tar $SCRATCH/ollama_base.tar.zst

    # Download ROCm libs
    curl -fsSL https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64-rocm.tar.zst | \
        zstd -d | tar -xf - -C $SCRATCH/bin/

    # Link ROCm libs next to binary
    ln -sf $SCRATCH/bin/lib/ollama $SCRATCH/bin/bin/lib 2>/dev/null || true

    echo 'Ollama installed:'
    $SCRATCH/bin/bin/ollama --version
  fi
"

# ── Step 4: Pull model ────────────────────────────────────────────────────
echo
echo ">>> Step 4/6: Pulling gpt-oss:120b (65 GB — runs in background)..."
$SSH "
  export OLLAMA_MODELS=$SCRATCH/model/ollama
  export OLLAMA_HOST=127.0.0.1:11434

  if ls $SCRATCH/model/ollama/models/manifests/registry.ollama.ai/library/gpt-oss/ 2>/dev/null | grep -q 120b; then
    echo 'Model already downloaded.'
  else
    echo 'Starting download in background (check $SCRATCH/model/pull.log)...'
    $SCRATCH/bin/bin/ollama serve > /tmp/ollama_pull.log 2>&1 &
    sleep 5
    nohup bash -c \"
      export OLLAMA_MODELS=$SCRATCH/model/ollama
      export OLLAMA_HOST=127.0.0.1:11434
      $SCRATCH/bin/bin/ollama pull gpt-oss:120b
      echo DONE > $SCRATCH/model/pull_done.flag
    \" > $SCRATCH/model/pull.log 2>&1 &
    echo 'Download started. Monitor with: tail -f $SCRATCH/model/pull.log'
  fi
"

# ── Step 5: Python venv ───────────────────────────────────────────────────
echo
echo ">>> Step 5/6: Setting up Python venv..."
$SSH "
  if [ -f $SCRATCH/venv/bin/activate ]; then
    echo 'Venv already exists.'
  else
    python3 -m venv $SCRATCH/venv
    source $SCRATCH/venv/bin/activate
    pip install --quiet dspy-ai pandas scikit-learn tqdm
    pip install --quiet -e $SCRATCH/code/
    python -c 'import dspy, mlgenx; print(\"dspy:\", dspy.__version__, \"mlgenx: ok\")'
  fi
"

# ── Step 6: Pre-cache tool calls ──────────────────────────────────────────
echo
echo ">>> Step 6/6: Building mygene + STRING cache locally..."
if [ -f "$REPO/cache/string_cache.json" ]; then
    echo "Cache already built. Transferring..."
else
    echo "Running warm_cache.py (takes ~10 min)..."
    python "$REPO/scripts/warm_cache.py"
fi

rsync -av -e "ssh -i $SSH_KEY" \
    "$REPO/cache/" \
    "$LUMI_USER@$LUMI_HOST:$SCRATCH/code/cache/"

echo
echo "========================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Wait for model download:  ssh lumi 'tail -f $SCRATCH/model/pull.log'"
echo "  2. Run eval:                 ssh lumi 'sbatch $SCRATCH/code/slurm/run_track_b_eval.sh'"
echo "  3. Full run:                 ssh lumi 'sbatch $SCRATCH/code/slurm/run_track_b.sh'"
echo "  4. Get results:              rsync -av lumi:$SCRATCH/outputs/track_b/ outputs/track_b_120b/"
echo "========================================"
