"""
Track B submission agent — DSPy ReAct loop with 6 retrieval tools.

Usage (dev):
    python track_b_submit.py --model gpt-oss:20b --concurrency 4 --rows 16

Usage (full run):
    python track_b_submit.py --model gpt-oss:120b --concurrency 32

Evaluate on dev_val:
    python eval/eval_harness.py --cache outputs/track_b/<run>/responses_cache.json
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
import litellm
import numpy as np
import pandas as pd

from tools.gene_info               import gene_info
from tools.protein_interactions    import protein_interactions
from tools.train_data_lookup       import train_data_lookup
from tools.tf_target_edge          import tf_target_edge
from tools.perturb_seq_lookup      import perturb_seq_lookup
from tools.base_rates_and_examples import base_rates_and_examples

ROOT           = Path(__file__).resolve().parent
TEST_CSV       = ROOT / "data" / "test.csv"
TRAIN_CSV      = ROOT / "data" / "train.csv"
DEFAULT_PROMPT = ROOT / "prompts" / "prompt.txt"


# ---------------------------------------------------------------------------
# Task 7 — per-thread token accumulator via litellm callback
# ---------------------------------------------------------------------------

_thread_local = threading.local()
_first_prompt_tokens: dict[str, int | None] = {"value": None}


def _litellm_token_callback(kwargs, completion_response, start_time, end_time):
    usage: dict = {}
    if isinstance(completion_response, dict):
        usage = completion_response.get("usage", {}) or {}
    elif hasattr(completion_response, "usage") and completion_response.usage:
        u = completion_response.usage
        usage = {
            "prompt_tokens":     getattr(u, "prompt_tokens",     0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens":      getattr(u, "total_tokens",      0),
        }

    total = int(usage.get("total_tokens", 0) or 0)
    if not hasattr(_thread_local, "tokens"):
        _thread_local.tokens = 0
    _thread_local.tokens += total

    # Task 9 — capture real prompt_tokens from the first API call per thread
    pt = int(usage.get("prompt_tokens", 0) or 0)
    if pt > 0 and not getattr(_thread_local, "prompt_tokens_set", False):
        _thread_local.first_prompt_tokens = pt
        _thread_local.prompt_tokens_set = True
        if _first_prompt_tokens["value"] is None:
            _first_prompt_tokens["value"] = pt


litellm.success_callback = [_litellm_token_callback]


# ---------------------------------------------------------------------------
# submit_answer tool
# ---------------------------------------------------------------------------

_submit_local = threading.local()


def submit_answer(up: float, down: float) -> str:
    """Submit your final predicted probabilities for this gene pair.
    up   = probability that the target gene is UP-regulated (0.0 to 1.0).
    down = probability that the target gene is DOWN-regulated (0.0 to 1.0).
    The probability of no effect is implied: 1 - up - down.
    You MUST call this tool to record your answer."""
    try:
        up_f   = max(0.0, min(1.0, float(up)))
        down_f = max(0.0, min(1.0, float(down)))
        if up_f + down_f > 1.0:
            scale  = 0.99 / (up_f + down_f)
            up_f, down_f = up_f * scale, down_f * scale
        _submit_local.result = (up_f, down_f)
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
    question: str = dspy.InputField(desc="Gene expression prediction question")
    answer:   str = dspy.OutputField(
        desc="Final reasoning summary after calling submit_answer"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_from_trajectory(trajectory: dict) -> tuple[float, float] | None:
    """Task 8 — extract (up, down) from the last submit_answer call in the trajectory."""
    last = None
    for i in range(250):
        if f"tool_name_{i}" not in trajectory:
            break
        if trajectory[f"tool_name_{i}"] == "submit_answer":
            args = trajectory.get(f"tool_args_{i}", {})
            if isinstance(args, dict):
                try:
                    last = (float(args.get("up", 0)), float(args.get("down", 0)))
                except (TypeError, ValueError):
                    pass
    return last


def _extract_floats_fallback(text: str) -> tuple[float, float]:
    """Task 1 — last-resort text scan; returns graded prior, never hard 0/1."""
    m = re.search(r'"?up"?\s*[=:]\s*([0-9.]+)', text, re.IGNORECASE)
    n = re.search(r'"?down"?\s*[=:]\s*([0-9.]+)', text, re.IGNORECASE)
    if m and n:
        try:
            return float(m.group(1)), float(n.group(1))
        except ValueError:
            pass
    return 0.31, 0.14  # graded prior — never A/B/C hard values


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
    parser.add_argument("--api-base",  default="http://localhost:11434/v1")
    parser.add_argument("--api-key",   default="none")
    parser.add_argument("--model",     default="gpt-oss:20b")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--max-iters",   type=int, default=12,
                        help="Max ReAct iterations (tool calls) per row per sample")
    parser.add_argument("--samples",     type=int, default=1,
                        help="Agent samples per row; predictions are averaged (default 1)")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--rows",        type=int, default=None)
    parser.add_argument("--test-csv",  type=Path, default=TEST_CSV)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "track_b")
    parser.add_argument("--run-name",  default=None)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--save-every",  type=int, default=10)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--eval-train",  action="store_true",
                        help="Run on dev_val split of train.csv")
    parser.add_argument("--calibrate",   action="store_true",
                        help="Fit calibrators (eval-train) or apply (test, needs --calibrators-from)")
    parser.add_argument("--calibrators-from", type=Path, default=None,
                        help="Path to calibrators.joblib from a prior eval-train run")
    args = parser.parse_args()

    model_name = args.model_name or args.model

    # ── Experiment directory ────────────────────────────────────────────────
    run_name   = args.run_name or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name}  →  {output_dir}")

    # ── System prompt ───────────────────────────────────────────────────────
    system_prompt  = args.system_prompt.read_text().strip()
    rough_tokens   = len(system_prompt) // 4
    print(f"System prompt: {len(system_prompt)} chars (~{rough_tokens} tokens; "
          f"actual measured from first API call)")
    if rough_tokens > 4096:
        print(f"WARNING: prompt may approach 4,096 token limit")

    # ── Task 3 — DSPy LM with top_p=1.0 and reasoning_effort ───────────────
    lm = dspy.LM(
        model=f"openai/{args.model}",
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
        reasoning_effort=args.reasoning_effort,
        num_retries=2,
    )
    dspy.configure(
        lm=lm,
        adapter=dspy.ChatAdapter(use_native_function_calling=False),
    )

    # ── Task 11 — drop train_data_lookup for test runs ──────────────────────
    if args.eval_train:
        tool_list = [
            tf_target_edge, perturb_seq_lookup, gene_info,
            protein_interactions, train_data_lookup,
            base_rates_and_examples, submit_answer,
        ]
    else:
        # train_data_lookup cannot hit test pairs (splits are disjoint on both axes)
        tool_list = [
            tf_target_edge, perturb_seq_lookup, gene_info,
            protein_interactions, base_rates_and_examples, submit_answer,
        ]
    num_distinct_tools = len(tool_list)
    print(f"Tools: {num_distinct_tools}, max_iters: {args.max_iters}, "
          f"samples: {args.samples}, concurrency: {args.concurrency}")

    react = dspy.ReAct(BioPredict, tools=tool_list, max_iters=args.max_iters)

    # ── Load data ───────────────────────────────────────────────────────────
    if args.eval_train:
        from eval.eval_harness import make_split
        _, test_df = make_split()
        print(f"Running on dev_val ({len(test_df)} rows)")
    else:
        test_df = pd.read_csv(args.test_csv)

    if args.rows:
        test_df = test_df.head(args.rows)
        print(f"Limited to {len(test_df)} rows")

    # ── Cache ───────────────────────────────────────────────────────────────
    cache_path = output_dir / "responses_cache.json"
    if args.clear_cache and cache_path.exists():
        cache_path.unlink()
    cache = load_cache(cache_path)
    if "rows" not in cache:
        cache["rows"] = {}

    total      = len(test_df)
    cache_lock = threading.Lock()
    new_count  = 0

    def run_single_sample(rid: str, pert: str, gene: str) -> dict:
        """Run one agent sample; return a per-sample result dict."""
        # Task 1 — minimal question without A/B/C framing
        user_prompt = (
            f"Perturbed gene (CRISPRi knockdown): {pert}\n"
            f"Target gene: {gene}\n"
            "Gather evidence with the tools, then call submit_answer(up, down) "
            "with graded probabilities (use the full 0–1 range to express confidence)."
        )
        _submit_local.result = None
        _thread_local.tokens = 0
        _thread_local.prompt_tokens_set = False

        tool_calls_count = 0
        trace: Any = {}

        try:
            result = react(question=user_prompt)
            final_text = result.answer or ""
            trajectory  = getattr(result, "trajectory", {}) or {}
            trace = trajectory
            tool_calls_count = sum(
                1 for k in trajectory
                if isinstance(k, str) and k.startswith("tool_name")
            )
        except Exception as e:
            print(f"  [error] {rid}: {e}")
            final_text = ""
            trace = {"error": str(e)}

        tokens = _thread_local.tokens  # Task 7 — thread-safe count

        # Task 8 — prefer trajectory → thread-local → text fallback
        submitted = _result_from_trajectory(trace)
        if submitted is None:
            submitted = getattr(_submit_local, "result", None)
        if submitted is not None:
            pred_up, pred_down = submitted
            source = "llm"
        else:
            print(f"  [fallback] {rid}: submit_answer not called, using prior")
            pred_up, pred_down = _extract_floats_fallback(final_text)
            source = "fallback"

        return {
            "prediction_up":   pred_up,
            "prediction_down": pred_down,
            "reasoning_trace": json.dumps(trace, default=str)[:8000],
            "tokens_used":     tokens,
            "num_tool_calls":  tool_calls_count,
            "model_name":      model_name,
            "source":          source,
        }

    def process_row(idx: int, row: pd.Series) -> None:
        nonlocal new_count
        rid  = row["id"]
        pert = row["pert"]
        gene = row["gene"]

        # Task 6 — multi-sample cache check
        with cache_lock:
            cached = cache["rows"].get(rid, {})
            # Backward-compatible: old single-sample entries have no "samples" key
            if args.samples <= 1 and "prediction_up" in cached:
                return
            existing_samples = cached.get("samples", [])
            if len(existing_samples) >= args.samples and "prediction_up" in cached:
                return

        samples = list(existing_samples)
        for _ in range(len(samples), args.samples):
            samples.append(run_single_sample(rid, pert, gene))

        pred_up    = float(np.mean([s["prediction_up"]   for s in samples]))
        pred_down  = float(np.mean([s["prediction_down"] for s in samples]))
        tokens     = sum(s["tokens_used"]    for s in samples)
        tool_calls = sum(s["num_tool_calls"] for s in samples)
        source     = "fallback" if any(s["source"] == "fallback" for s in samples) else "llm"

        entry: dict[str, Any] = {
            "prediction_up":   pred_up,
            "prediction_down": pred_down,
            "reasoning_trace": samples[0]["reasoning_trace"],
            "tokens_used":     tokens,
            "num_tool_calls":  tool_calls,
            "model_name":      model_name,
            "source":          source,
        }
        if args.samples > 1:
            entry["samples"] = samples

        with cache_lock:
            cache["rows"][rid] = entry
            new_count += 1
            print(
                f"[{idx+1}/{total}] {rid}  "
                f"up={pred_up:.3f} down={pred_down:.3f}  "
                f"tools={tool_calls} tok={tokens} src={source}"
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

    # ── Task 9 — actual prompt token count from first API call ─────────────
    prompt_tokens = _first_prompt_tokens["value"] or rough_tokens
    print(f"Prompt tokens (actual): {prompt_tokens}")
    if prompt_tokens > 16384:
        print(f"WARNING: prompt_tokens {prompt_tokens} exceeds 16,384 limit!")

    # ── Calibration ─────────────────────────────────────────────────────────
    cal_de = cal_dir = None
    if args.calibrate:
        from eval.calibrate import fit_calibrators, save_calibrators, load_calibrators
        if args.eval_train:
            from eval.eval_harness import make_split as _make_split
            _, dev_val_cal = _make_split()
            cal_de, cal_dir = fit_calibrators(dev_val_cal, cache["rows"])
            cal_path = output_dir / "calibrators.joblib"
            save_calibrators(cal_de, cal_dir, cal_path)
            print(f"Calibrators saved → {cal_path}")
        elif args.calibrators_from:
            cal_de, cal_dir = load_calibrators(args.calibrators_from)
            print(f"Calibrators loaded from {args.calibrators_from}")
        else:
            print("WARNING: --calibrate set but no labels to fit on; "
                  "pass --calibrators-from <path> for test runs")

    # ── Task 10 — eval-train: print metrics and exit, no test zip ──────────
    if args.eval_train:
        from eval.eval_harness import evaluate, print_metrics, make_split as _ms
        _, dev_val = _ms()
        results = {rid: cache["rows"][rid]
                   for rid in cache["rows"] if rid in set(dev_val["id"])}
        print("\n=== dev_val metrics (raw) ===")
        print_metrics(evaluate(dev_val, results))
        if cal_de is not None:
            from eval.calibrate import apply_calibrators
            calibrated = {
                rid: {**r,
                      **dict(zip(("prediction_up", "prediction_down"),
                                 apply_calibrators(r["prediction_up"],
                                                   r["prediction_down"],
                                                   cal_de, cal_dir)))}
                for rid, r in results.items()
            }
            print("=== dev_val metrics (calibrated) ===")
            print_metrics(evaluate(dev_val, calibrated))
        return

    # ── Build submission CSV ─────────────────────────────────────────────────
    rows_out = []
    for _, row in pd.read_csv(args.test_csv).iterrows():
        rid = row["id"]
        c   = cache["rows"].get(rid, {})
        pred_up   = c.get("prediction_up",  0.31)
        pred_down = c.get("prediction_down", 0.14)
        if cal_de is not None:
            from eval.calibrate import apply_calibrators
            pred_up, pred_down = apply_calibrators(pred_up, pred_down, cal_de, cal_dir)
        rows_out.append({
            "id":                 rid,
            "prediction_up":      pred_up,
            "prediction_down":    pred_down,
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

    # ── Task 2 — package zip with correct tools/ path ───────────────────────
    prompt_out = output_dir / "prompt.txt"
    prompt_out.write_text(system_prompt)

    src_tools = Path(__file__).resolve().parent / "tools"
    tools_out = output_dir / "tools"
    if tools_out.exists():
        shutil.rmtree(tools_out)
    tools_out.mkdir()
    tool_names = {fn.__name__ + ".py" for fn in tool_list if fn is not submit_answer}
    for fname in tool_names:
        src = src_tools / fname
        if src.exists():
            shutil.copy2(src, tools_out / fname)
    init_src = src_tools / "__init__.py"
    if init_src.exists():
        shutil.copy2(init_src, tools_out / "__init__.py")

    zip_path = output_dir / "submission_track_b.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path,   "submission.csv")
        zf.write(prompt_out, "prompt.txt")
        for tool_file in sorted(tools_out.glob("*.py")):
            zf.write(tool_file, f"tools/{tool_file.name}")
    print(f"Saved {zip_path}  ← upload to Kaggle")


if __name__ == "__main__":
    main()
