"""
base_rates_and_examples -- Training priors and calibration examples.

Returns:
  - Global class priors (P(none), P(up), P(down)) from train.csv
  - Optional few-shot examples for a given perturbation gene
    (only from train — NEVER from test, splits are disjoint on perts)

Note: train.csv perts and test.csv perts are completely disjoint.
      This tool will find training examples for the same perturbed gene
      only when that gene appears in train (never in test predictions).
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import pandas as pd

TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "train.csv"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "base_rates_and_examples",
        "description": (
            "Return global class priors from training data and optionally "
            "a few labeled examples for the same perturbation gene "
            "(only available for train-set perts). "
            "Use priors to calibrate your confidence when evidence is weak."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pert": {
                    "type": "string",
                    "description": "Perturbation gene (optional — omit for priors only).",
                },
            },
        },
    },
}


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def base_rates_and_examples(pert: str | None = None) -> str:
    df = _load()
    total = len(df)
    counts = df["label"].value_counts()
    p_none = counts.get("none", 0) / total
    p_up   = counts.get("up",   0) / total
    p_down = counts.get("down", 0) / total

    lines = [
        "Global priors from training data (mouse BMDM CRISPRi screen):",
        f"  P(none) = {p_none:.3f}  P(up) = {p_up:.3f}  P(down) = {p_down:.3f}",
        f"  Among DE-positive: P(up|DE) = {p_up/(p_up+p_down):.3f}",
    ]

    if pert:
        hits = df[df["pert"].str.lower() == pert.lower()]
        if hits.empty:
            lines.append(
                f"\nNote: '{pert}' is not in training data — "
                f"it is a test-set perturbation. Rely on priors and tools."
            )
        else:
            up_genes   = hits[hits["label"] == "up"]["gene"].tolist()
            down_genes = hits[hits["label"] == "down"]["gene"].tolist()
            none_genes = hits[hits["label"] == "none"]["gene"].tolist()
            lines.append(
                f"\nTraining examples for pert='{pert}' ({len(hits)} rows): "
                f"{len(up_genes)} up, {len(down_genes)} down, {len(none_genes)} none."
            )
            if up_genes:
                lines.append(f"  Up-regulated:   {', '.join(up_genes[:20])}")
            if down_genes:
                lines.append(f"  Down-regulated: {', '.join(down_genes[:20])}")

    return "\n".join(lines)
