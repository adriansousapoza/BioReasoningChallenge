#!/bin/bash -l
# Slurm job: 100-row train evaluation on dev_val split.
# Use dev-g partition for fast queue access.
# Edit SCRATCH/PROJECT if needed.
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
OLLAMA=$SCRATCH/bin/bin/ollama

source $SCRATCH/venv/bin/activate
export OLLAMA_MODELS=$SCRATCH/model/ollama
export OLLAMA_HOST=127.0.0.1:11434
export BIOREASONDATA=$SCRATCH/data
export GRN_DATA_DIR=$SCRATCH/data/grn
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH=$SCRATCH/bin/bin:$PATH
mkdir -p $OUTPUT

echo "[$(date)] Node: $(hostname)"
echo "[$(date)] Starting Ollama (ROCm)..."
$OLLAMA serve &
OLLAMA_PID=$!

for i in $(seq 1 120); do
    $OLLAMA list > /dev/null 2>&1 && echo "[$(date)] Ollama ready (${i}x5s)" && break
    sleep 5
    [ $i -eq 120 ] && echo "ERROR: Ollama timeout" && kill $OLLAMA_PID && exit 1
done

curl -s http://127.0.0.1:11434/api/generate \
    -d '{"model":"gpt-oss:120b","prompt":"Hi","stream":false,"options":{"num_predict":5}}' > /dev/null
echo "[$(date)] Model loaded. Running 100-row eval (concurrency=16)..."

python $CODE/track_b_submit.py \
    --api-base http://127.0.0.1:11434/v1 \
    --api-key  ollama \
    --model    gpt-oss:120b \
    --model-name openai/gpt-oss-120b \
    --output-dir $OUTPUT \
    --run-name "${SLURM_JOB_ID}_eval" \
    --concurrency 16 \
    --max-iters 12 \
    --reasoning-effort medium \
    --save-every 20 \
    --eval-train \
    --rows 100 \
    --clear-cache

kill $OLLAMA_PID
echo "[$(date)] Done. Check $OUTPUT/"
