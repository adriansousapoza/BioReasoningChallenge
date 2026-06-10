# eval/

Evaluation utilities: local dev harness and the official Kaggle scoring metrics.

## Files

### `eval_harness.py`

Local validation harness for Track B. Creates a disjoint dev split from `data/train.csv` (split by perturbation gene so no pert appears in both dev_train and dev_val), runs predictions through the official metric, and prints scores.

```bash
# Print split statistics
python eval/eval_harness.py --show-split

# Score a finished submission CSV
python eval/eval_harness.py --submission outputs/track_b/2026-06-10_143022/submission.csv

# Score directly from a responses cache (no need to generate CSV first)
python eval/eval_harness.py --cache outputs/track_b/2026-06-10_143022/responses_cache.json
```

Also imported by `track_b_submit.py` when running with `--eval-train`.

### `kaggle_metric.py`

Base metric implementation: computes the average of **DE AUROC** (up+down vs none) and **DIR AUROC** (up vs down, among DE-positive rows only). Follows Kaggle's metric API (`score(solution, submission, row_id_column_name)`).

### `kaggle_metric_track_a.py` / `_track_b.py` / `_track_c.py`

Track-specific metric wrappers. Each enforces the column requirements and constraints for its track (e.g. Track B checks `num_tool_calls ≤ 250`, `prompt_tokens ≤ 16384`, `num_distinct_tools ≤ 100`). These are the files submitted to Kaggle as the official scoring metric.
