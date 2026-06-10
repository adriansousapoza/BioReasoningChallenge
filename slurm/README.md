# slurm/

SLURM batch scripts for running inference on [LUMI](https://lumi-supercomputer.eu/) (8× AMD MI250X GPUs, standard-g partition).

All scripts expect the code to be at `$SCRATCH/code/` on LUMI. See `scripts/setup_syncthing_lumi.sh` for how to sync code to LUMI.

## Scripts

### `run_track_b.sh` — full test run

Runs all 1,813 test rows. Requests an 8-hour reservation on `standard-g`.

```bash
ssh lumi-uan01 "sbatch /scratch/project_465002610/sousapoz/code/slurm/run_track_b.sh"
```

Expected runtime: ~6–8 hours. Output goes to `outputs/track_b/{SLURM_JOB_ID}_full/`.

### `run_track_b_eval.sh` — 100-row dev evaluation

Runs 100 rows from the training split (dev_val) for quick feedback. Uses the faster `dev-g` partition (shorter queue).

```bash
ssh lumi-uan01 "sbatch /scratch/project_465002610/sousapoz/code/slurm/run_track_b_eval.sh"
```

Expected runtime: ~30–45 minutes. Output goes to `outputs/track_b/{SLURM_JOB_ID}_eval/`.

### `run_track_b_resume.sh` — resume an interrupted run

Picks up from an existing `responses_cache.json`, skipping already-completed rows. Hardcoded to resume run `19134972_full`; edit `--run-name` to resume a different run.

```bash
ssh lumi-uan01 "sbatch /scratch/project_465002610/sousapoz/code/slurm/run_track_b_resume.sh"
```

## Checking job status

```bash
ssh lumi-uan01 "squeue -u sousapoz"
ssh lumi-uan01 "tail -f /scratch/project_465002610/sousapoz/code/outputs/track_b/bioreasonB_<JOBID>.out"
```
