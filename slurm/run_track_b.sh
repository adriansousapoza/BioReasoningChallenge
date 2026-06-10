#!/bin/bash -l
# Full test.csv run — vLLM with tensor parallelism across all 8 MI250X GCDs.
#
# Prerequisites (run once before submitting):
#   cd $SCRATCH/code && uv sync --extra serve   # installs vLLM into .venv
#   huggingface-cli download openai/gpt-oss-120b --local-dir $SCRATCH/model/hf/gpt-oss-120b
#
# Expected: ~3-4 hours for 1813 rows at concurrency=32 with full GPU utilisation.
#SBATCH --job-name=bioreasonB
#SBATCH --output=/scratch/project_465002610/sousapoz/code/outputs/track_b/%x_%j.out
#SBATCH --error=/scratch/project_465002610/sousapoz/code/outputs/track_b/%x_%j.err
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --account=project_465002610

set -e
SCRATCH=/scratch/project_465002610/sousapoz
CODE=$SCRATCH/code
OUTPUT=$CODE/outputs/track_b
MODEL=$SCRATCH/model/hf/gpt-oss-120b

# Syncthing runs only in a tmux session on the login node — it is NOT present
# on compute nodes. This kill is a no-op on the compute node but serves as an
# explicit safeguard in case anything ever starts it here accidentally.
pkill -x syncthing || true

source $CODE/.venv/bin/activate
export HF_HOME=$SCRATCH/model/hf
export BIOREASONDATA=$SCRATCH/data
export GRN_DATA_DIR=$SCRATCH/data/grn
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p $OUTPUT

echo "[$(date)] Node: $(hostname), partition: standard-g, 8 GCDs"
echo "[$(date)] Starting vLLM (ROCm, tensor-parallel=8)..."

python -m vllm.entrypoints.openai.api_server \
    $MODEL \
    --served-model-name gpt-oss:120b \
    --tensor-parallel-size 8 \
    --host 127.0.0.1 \
    --port 11434 \
    --dtype auto \
    --enforce-eager \
    --no-enable-prefix-caching &
VLLM_PID=$!

# Wait for vLLM to be ready (health endpoint)
for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:11434/health > /dev/null 2>&1 \
        && echo "[$(date)] vLLM ready (${i}x5s)" && break
    sleep 5
    [ $i -eq 120 ] && echo "ERROR: vLLM timeout" && kill $VLLM_PID && exit 1
done

# Warm up: load all tensor-parallel shards into VRAM
echo "[$(date)] Warming up model across all 8 GCDs..."
curl -sf http://127.0.0.1:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss:120b","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' \
    > /dev/null || true
echo "[$(date)] Model warm. Starting predictions (concurrency=32)..."

python $CODE/track_b_submit.py \
    --api-base    http://127.0.0.1:11434/v1 \
    --api-key     none \
    --model       gpt-oss:120b \
    --model-name  openai/gpt-oss-120b \
    --test-csv    $CODE/data/test.csv \
    --output-dir  $OUTPUT \
    --run-name    "${SLURM_JOB_ID}_full" \
    --concurrency 32 \
    --max-iters   12 \
    --reasoning-effort medium \
    --save-every  50

kill $VLLM_PID
echo "[$(date)] Done. Results at $OUTPUT/${SLURM_JOB_ID}_full/"
