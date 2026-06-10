#!/bin/bash -l
# Resume job: continues 19134972_full with remaining rows from cache.
#SBATCH --job-name=bioreasonB_resume
#SBATCH --output=/scratch/project_465002610/sousapoz/code/outputs/track_b/bioreasonB_resume_%j.out
#SBATCH --error=/scratch/project_465002610/sousapoz/code/outputs/track_b/bioreasonB_resume_%j.err
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
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
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_FLASH_ATTENTION=1
export PATH=$SCRATCH/bin/bin:$PATH
mkdir -p $OUTPUT

echo "[$(date)] Node: $(hostname), resuming run 19134972_full (363 rows remaining)"

$OLLAMA serve &
OLLAMA_PID=$!

for i in $(seq 1 120); do
    $OLLAMA list > /dev/null 2>&1 && echo "[$(date)] Ollama ready (${i}x5s)" && break
    sleep 5
    [ $i -eq 120 ] && echo "ERROR: Ollama timeout" && kill $OLLAMA_PID && exit 1
done

curl -sf http://127.0.0.1:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss:120b","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' > /dev/null || true
echo "[$(date)] Model warm. Resuming remaining rows (cache: 19134972_full)..."

python $CODE/track_b_submit.py \
    --api-base http://127.0.0.1:11434/v1 \
    --api-key  ollama \
    --model    gpt-oss:120b \
    --model-name openai/gpt-oss-120b \
    --test-csv $CODE/data/test.csv \
    --output-dir $OUTPUT \
    --run-name 19134972_full \
    --concurrency 16 \
    --max-iters 12 \
    --reasoning-effort medium \
    --save-every 50

kill $OLLAMA_PID
echo "[$(date)] Done. Check $OUTPUT/19134972_full/"
