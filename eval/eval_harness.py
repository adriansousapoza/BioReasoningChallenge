"""
Local validation harness for Track B.

Splits train.csv by perturbation gene (disjoint perts, disjoint genes)
into dev_train / dev_val, then evaluates predictions with the official metric.

Usage:
    # print the split stats
    python eval/eval_harness.py --show-split

    # score a submission CSV
    python eval/eval_harness.py --submission outputs/track_b/<run>/submission.csv

    # evaluate a predictions JSON cache
    python eval/eval_harness.py --cache outputs/track_b/<run>/responses_cache.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT     = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "data" / "train.csv"


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def make_split(seed: int = 42, val_frac: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train.csv into dev_train / dev_val disjoint on BOTH perturbation AND target genes.

    Perts and genes are partitioned independently (seeded), then we take the
    doubly-disjoint quadrants:
      dev_val   = val_perts   × val_genes
      dev_train = train_perts × train_genes  (no val_genes)

    This matches the real test condition where perts and target genes are both unseen.
    Returns (dev_train, dev_val).
    """
    df = pd.read_csv(TRAIN_CSV)
    rng = random.Random(seed)

    perts = df["pert"].unique().tolist()
    rng.shuffle(perts)
    n_val_p  = max(1, int(len(perts) * val_frac))
    val_perts   = set(perts[:n_val_p])
    train_perts = set(perts[n_val_p:])

    genes = df["gene"].unique().tolist()
    rng.shuffle(genes)
    n_val_g  = max(1, int(len(genes) * val_frac))
    val_genes = set(genes[:n_val_g])

    dev_val   = df[df["pert"].isin(val_perts)   &  df["gene"].isin(val_genes)].reset_index(drop=True)
    dev_train = df[df["pert"].isin(train_perts)  & ~df["gene"].isin(val_genes)].reset_index(drop=True)
    return dev_train, dev_val


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(
    labels: pd.Series,
    pred_up: np.ndarray,
    pred_down: np.ndarray,
) -> dict:
    labels = labels.values
    y_de = (labels != "none").astype(int)
    s_de = pred_up + pred_down
    auroc_de = roc_auc_score(y_de, s_de)

    de_mask = labels != "none"
    y_dir  = (labels[de_mask] == "up").astype(int)
    denom  = (pred_up[de_mask] + pred_down[de_mask])
    denom  = np.where(denom == 0, 1.0, denom)
    s_dir  = pred_up[de_mask] / denom
    auroc_dir = roc_auc_score(y_dir, s_dir)

    return {
        "de_auroc":  round(auroc_de,  4),
        "dir_auroc": round(auroc_dir, 4),
        "score":     round((auroc_de + auroc_dir) / 2, 4),
        "n_total":   int(len(labels)),
        "n_de":      int(de_mask.sum()),
    }


def evaluate(df: pd.DataFrame, results: dict[str, dict]) -> dict:
    """
    Evaluate a results dict {id: {prediction_up, prediction_down, source?}}
    against a labeled dataframe.  Missing rows default to (0, 0).
    """
    rows = []
    for _, row in df.iterrows():
        r = results.get(row["id"], {"prediction_up": 0.0, "prediction_down": 0.0})
        rows.append({
            "label": row["label"],
            "up": float(r.get("prediction_up", 0.0)),
            "down": float(r.get("prediction_down", 0.0)),
            "source": r.get("source", "unknown"),
        })
    pred = pd.DataFrame(rows)

    metrics = compute_score(pred["label"], pred["up"].values, pred["down"].values)
    metrics["n_predicted"] = sum(1 for r in results.values() if r.get("prediction_up") is not None)

    # Per-source breakdown
    for src, grp in pred.groupby("source"):
        if len(grp) < 3:
            continue
        m = compute_score(grp["label"], grp["up"].values, grp["down"].values)
        metrics[f"source_{src}"] = m

    return metrics


def print_metrics(metrics: dict, prefix: str = "") -> None:
    p = f"{prefix} " if prefix else ""
    print(f"{p}DE-AUROC={metrics['de_auroc']:.4f}  "
          f"DIR-AUROC={metrics['dir_auroc']:.4f}  "
          f"Score={metrics['score']:.4f}  "
          f"(n={metrics['n_total']}, DE={metrics['n_de']})")
    for k, v in metrics.items():
        if k.startswith("source_") and isinstance(v, dict):
            print(f"  [{k[7:]}]  DE={v['de_auroc']:.3f}  DIR={v['dir_auroc']:.3f}  "
                  f"Score={v['score']:.3f}  n={v['n_total']}")


# ---------------------------------------------------------------------------
# Smoke set
# ---------------------------------------------------------------------------

SMOKE_IDS = None

def get_smoke_set(n: int = 16) -> pd.DataFrame:
    """Fixed small set spanning up/down/none for sub-minute iteration."""
    global SMOKE_IDS
    df = pd.read_csv(TRAIN_CSV)
    rng = random.Random(0)
    rows = []
    for label in ["up", "down", "none"]:
        pool = df[df["label"] == label].to_dict("records")
        rows.extend(rng.sample(pool, min(n // 3, len(pool))))
    smoke = pd.DataFrame(rows).head(n).reset_index(drop=True)
    SMOKE_IDS = set(smoke["id"])
    return smoke


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-split", action="store_true")
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Score smoke set only")
    args = parser.parse_args()

    dev_train, dev_val = make_split()

    if args.show_split:
        print(f"dev_train: {len(dev_train)} rows, {dev_train['pert'].nunique()} perts, "
              f"{dev_train['gene'].nunique()} genes")
        print(f"dev_val:   {len(dev_val)} rows, {dev_val['pert'].nunique()} perts, "
              f"{dev_val['gene'].nunique()} genes")
        gene_overlap = set(dev_train["gene"]) & set(dev_val["gene"])
        print(f"Gene overlap: {len(gene_overlap)}")
        print(f"Label dist dev_val: {dev_val['label'].value_counts().to_dict()}")
        return

    eval_df = get_smoke_set() if args.smoke else dev_val

    if args.submission:
        sub = pd.read_csv(args.submission)
        results = {
            r["id"]: {"prediction_up": r["prediction_up"], "prediction_down": r["prediction_down"]}
            for _, r in sub.iterrows()
        }
    elif args.cache:
        with open(args.cache) as f:
            cache = json.load(f)
        results = cache.get("rows", cache)
    else:
        parser.print_help()
        return

    metrics = evaluate(eval_df, results)
    print_metrics(metrics, prefix="dev_val" if not args.smoke else "smoke")


if __name__ == "__main__":
    main()
