#!/bin/bash -l
# 100-row train evaluation on dev_val split — vLLM with tensor parallelism.
# Uses dev-g partition for fast queue access (~30-45 min turnaround).
#
# Prerequisites (run once before submitting):
#   cd $SCRATCH/code && uv sync --extra serve
#   huggingface-cli download openai/gpt-oss-120b --local-dir $SCRATCH/model/hf/gpt-oss-120b
#SBATCH --job-name=bioreasonB_eval
#SBATCH --output=/scratch/project_465002610/sousapoz/code/outputs/track_b/eval_%j.out
#SBATCH --error=/scratch/project_465002610/sousapoz/code/outputs/track_b/eval_%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --account=project_465002610

set -e
SCRATCH=/scratch/project_465002610/sousapoz
CODE=$SCRATCH/code
OUTPUT=$CODE/outputs/track_b
MODEL=$SCRATCH/model/hf/gpt-oss-120b

pkill -x syncthing || true

source $CODE/.venv/bin/activate
export HF_HOME=$SCRATCH/model/hf
export BIOREASONDATA=$SCRATCH/data
export GRN_DATA_DIR=$SCRATCH/data/grn
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p $OUTPUT

echo "[$(date)] Node: $(hostname), starting vLLM (tensor-parallel=8)..."

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

for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:11434/health > /dev/null 2>&1 \
        && echo "[$(date)] vLLM ready (${i}x5s)" && break
    sleep 5
    [ $i -eq 120 ] && echo "ERROR: vLLM timeout" && kill $VLLM_PID && exit 1
done

curl -sf http://127.0.0.1:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss:120b","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' \
    > /dev/null || true
echo "[$(date)] Model warm. Running 100-row eval (concurrency=32)..."

python $CODE/track_b_submit.py \
    --api-base    http://127.0.0.1:11434/v1 \
    --api-key     none \
    --model       gpt-oss:120b \
    --model-name  openai/gpt-oss-120b \
    --output-dir  $OUTPUT \
    --run-name    "${SLURM_JOB_ID}_eval" \
    --concurrency 32 \
    --max-iters   12 \
    --reasoning-effort medium \
    --save-every  20 \
    --eval-train \
    --rows 100 \
    --clear-cache

kill $VLLM_PID
echo "[$(date)] Done. Check $OUTPUT/${SLURM_JOB_ID}_eval/"
