"""
Track B submission agent — extends the official example with:
  - tf_target_edge, perturb_seq_lookup, base_rates_and_examples tools
  - submit_answer accepts float (up, down) probabilities instead of A/B/C
  - Caching / resumability
  - Local eval harness integration
  - Calibration post-processing

Usage (dev with 20b):
    python track_b_submit.py --model gpt-oss:20b --concurrency 4 --rows 50

Usage (full 120b run):
    python track_b_submit.py --model openai/gpt-oss-120b \
        --api-base http://localhost:8000/v1 --concurrency 8

Evaluate on dev_val:
    python eval/eval_harness.py --cache outputs/track_b/responses_cache.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
import pandas as pd

from mlgenx import format_prompt
from mlgenx.prompts import CELL_DESC, _PROMPT_ZERO

# ── local tools ──────────────────────────────────────────────────────────
from tools.gene_info            import gene_info
from tools.protein_interactions import protein_interactions
from tools.train_data_lookup    import train_data_lookup
from tools.tf_target_edge       import tf_target_edge
from tools.perturb_seq_lookup   import perturb_seq_lookup
from tools.base_rates_and_examples import base_rates_and_examples

ROOT     = Path(__file__).resolve().parent
TEST_CSV  = ROOT / "data" / "test.csv"
TRAIN_CSV = ROOT / "data" / "train.csv"
DEFAULT_PROMPT = ROOT / "prompts" / "prompt.txt"


# ---------------------------------------------------------------------------
# submit_answer — accepts float probabilities
# ---------------------------------------------------------------------------

_submit_lock = threading.local()


def submit_answer(up: float, down: float) -> str:
    """Submit your final predicted probabilities for this gene pair.
    up   = probability that the target gene is UP-regulated (0.0 to 1.0).
    down = probability that the target gene is DOWN-regulated (0.0 to 1.0).
    The probability of no effect is implied: 1 - up - down.
    You MUST call this tool to record your answer."""
    try:
        up_f   = float(up)
        down_f = float(down)
        up_f   = max(0.0, min(1.0, up_f))
        down_f = max(0.0, min(1.0, down_f))
        if up_f + down_f > 1.0:
            scale  = 0.99 / (up_f + down_f)
            up_f, down_f = up_f * scale, down_f * scale
        # Store in thread-local so the caller can retrieve it
        _submit_lock.result = (up_f, down_f)
        return f"Answer recorded: up={up_f:.3f}, down={down_f:.3f}, none={1-up_f-down_f:.3f}"
    except Exception as e:
        return f"Error parsing probabilities: {e}. Provide floats between 0 and 1."


# ---------------------------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------------------------

class BioPredict(dspy.Signature):
    """Predict the effect of a CRISPRi knockdown on a target gene in mouse BMDMs.
    Use tools to gather evidence, then call submit_answer(up, down) with your
    probability estimates."""

    question: str = dspy.InputField(
        desc="Gene expression prediction question"
    )
    answer: str = dspy.OutputField(
        desc="Final reasoning summary after calling submit_answer"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens_from_history(lm: dspy.LM, start_idx: int) -> int:
    total = 0
    for entry in lm.history[start_idx:]:
        if isinstance(entry, dict):
            usage = entry.get("usage") or {}
            if isinstance(usage, dict):
                total += usage.get("total_tokens", 0)
            else:
                total += getattr(usage, "total_tokens", 0) or 0
        elif hasattr(entry, "usage"):
            u = entry.usage
            total += getattr(u, "total_tokens", 0) if u else 0
    return total


def _extract_floats_from_text(text: str) -> tuple[float, float] | None:
    """Fallback: parse up/down from free text if submit_answer wasn't called."""
    m = re.search(r'"?up"?\s*[=:]\s*([0-9.]+)', text, re.IGNORECASE)
    n = re.search(r'"?down"?\s*[=:]\s*([0-9.]+)', text, re.IGNORECASE)
    if m and n:
        try:
            return float(m.group(1)), float(n.group(1))
        except ValueError:
            pass
    # Fall back to A/B/C parsing
    from mlgenx import parse_answer
    return parse_answer(text)


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_cache(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Track B submission agent")
    parser.add_argument("--api-base", default="http://localhost:11434/v1",
                        help="OpenAI-compatible endpoint (Ollama default)")
    parser.add_argument("--api-key",  default="ollama")
    parser.add_argument("--model",    default="gpt-oss:20b")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--max-iters", type=int, default=12,
                        help="Max ReAct iterations (tool calls) per row")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--rows", type=int, default=None,
                        help="Limit to first N rows (for dev/testing)")
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "track_b",
                        help="Base output directory; experiments go in a subdirectory")
    parser.add_argument("--run-name", default=None,
                        help="Experiment name (default: YYYY-MM-DD_HHMMSS). "
                             "Slurm scripts pass the job ID here.")
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--eval-train", action="store_true",
                        help="Run on dev_val split of train.csv instead of test.csv")
    args = parser.parse_args()

    model_name = args.model_name or args.model

    # ── Experiment directory ───────────────────────────────────────────
    run_name   = args.run_name or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name}  →  {output_dir}")

    # ── System prompt ─────────────────────────────────────────────────
    system_prompt = args.system_prompt.read_text().strip()
    prompt_tokens = len(system_prompt) // 4  # rough estimate
    print(f"System prompt: ~{prompt_tokens} tokens")
    if prompt_tokens > 4096:
        print(f"WARNING: prompt ~{prompt_tokens} tokens may exceed 4,096 limit")

    # ── Configure DSPy + Ollama ───────────────────────────────────────
    # Ollama exposes OpenAI-compatible /v1 endpoint
    lm = dspy.LM(
        model=f"openai/{args.model}",
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=1.0,
        num_retries=2,
    )
    dspy.configure(
        lm=lm,
        adapter=dspy.ChatAdapter(use_native_function_calling=False),
    )

    tool_list = [
        tf_target_edge,
        perturb_seq_lookup,
        gene_info,
        protein_interactions,
        train_data_lookup,
        base_rates_and_examples,
        submit_answer,
    ]
    num_distinct_tools = len(tool_list)
    print(f"Tools: {num_distinct_tools}, max_iters: {args.max_iters}")

    react = dspy.ReAct(BioPredict, tools=tool_list, max_iters=args.max_iters)

    # ── Load data ─────────────────────────────────────────────────────
    if args.eval_train:
        from eval.eval_harness import make_split
        _, test_df = make_split()
        print(f"Running on dev_val ({len(test_df)} rows)")
    else:
        test_df = pd.read_csv(args.test_csv)

    if args.rows:
        test_df = test_df.head(args.rows)
        print(f"Limited to {len(test_df)} rows")

    # ── Cache ─────────────────────────────────────────────────────────
    cache_path = output_dir / "responses_cache.json"
    if args.clear_cache and cache_path.exists():
        cache_path.unlink()
    cache = load_cache(cache_path)
    if "rows" not in cache:
        cache["rows"] = {}

    total     = len(test_df)
    cache_lock = threading.Lock()
    new_count  = 0

    def process_row(idx: int, row: pd.Series) -> None:
        nonlocal new_count
        rid = row["id"]

        with cache_lock:
            if rid in cache["rows"] and "prediction_up" in cache["rows"][rid]:
                return

        user_prompt = format_prompt(row["pert"], row["gene"])
        _submit_lock.result = None

        history_before = len(lm.history)
        tool_calls_count = 0
        trace: Any = {}

        try:
            result = react(question=user_prompt)
            final_text = result.answer or ""
            trajectory = getattr(result, "trajectory", {}) or {}
            trace = trajectory
            tool_calls_count = sum(
                1 for k in trajectory
                if isinstance(k, str) and k.startswith("tool_name")
            )
        except Exception as e:
            print(f"  [error] {rid}: {e}")
            final_text = ""
            trace = {"error": str(e)}

        tokens = _tokens_from_history(lm, history_before)

        # Prefer explicit submit_answer result; fall back to text parsing
        submitted = getattr(_submit_lock, "result", None)
        if submitted is not None:
            pred_up, pred_down = submitted
        else:
            pred_up, pred_down = _extract_floats_from_text(final_text)

        with cache_lock:
            cache["rows"][rid] = {
                "prediction_up":   pred_up,
                "prediction_down": pred_down,
                "reasoning_trace": json.dumps(trace, default=str)[:8000],
                "tokens_used":     tokens,
                "num_tool_calls":  tool_calls_count,
                "model_name":      model_name,
                "source":          "llm",
            }
            new_count += 1
            print(
                f"[{idx+1}/{total}] {rid}  "
                f"up={pred_up:.3f} down={pred_down:.3f}  "
                f"tools={tool_calls_count} tokens={tokens}"
            )
            if new_count % args.save_every == 0:
                save_cache(cache_path, cache)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(process_row, i, row)
                   for i, (_, row) in enumerate(test_df.iterrows())]
        for f in as_completed(futures):
            f.result()

    save_cache(cache_path, cache)
    print(f"\nDone. {total} rows, {new_count} new API calls.")

    # ── Build submission CSV ──────────────────────────────────────────
    rows_out = []
    for _, row in pd.read_csv(args.test_csv).iterrows():
        rid = row["id"]
        c   = cache["rows"].get(rid, {})
        rows_out.append({
            "id":                 rid,
            "prediction_up":      c.get("prediction_up",   0.31),
            "prediction_down":    c.get("prediction_down",  0.14),
            "reasoning_trace":    c.get("reasoning_trace",  "none"),
            "tokens_used":        int(c.get("tokens_used",  0)),
            "num_tool_calls":     int(c.get("num_tool_calls", 0)),
            "prompt_tokens":      prompt_tokens,
            "num_distinct_tools": num_distinct_tools,
            "model_name":         c.get("model_name", model_name),
        })

    sub_df   = pd.DataFrame(rows_out)
    sub_path = output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved {sub_path}")

    # ── Quick local eval if labels available (eval_train mode) ────────
    if args.eval_train:
        from eval.eval_harness import evaluate, print_metrics
        _, dev_val = make_split()
        results = {rid: cache["rows"][rid]
                   for rid in cache["rows"] if rid in set(dev_val["id"])}
        metrics = evaluate(dev_val, results)
        print("\n=== dev_val metrics ===")
        print_metrics(metrics)

    # ── Package zip ───────────────────────────────────────────────────
    prompt_out = output_dir / "prompt.txt"
    prompt_out.write_text(system_prompt)

    tools_out = output_dir / "tools"
    if tools_out.exists():
        shutil.rmtree(tools_out)
    src_tools = Path(__file__).resolve().parent / "examples" / "tools"
    shutil.copytree(src_tools, tools_out)

    zip_path = output_dir / "submission_track_b.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path,   "submission.csv")
        zf.write(prompt_out, "prompt.txt")
        for tool_file in tools_out.rglob("*.py"):
            zf.write(tool_file, f"tools/{tool_file.name}")
    print(f"Saved {zip_path}  ← upload to Kaggle")


if __name__ == "__main__":
    main()
