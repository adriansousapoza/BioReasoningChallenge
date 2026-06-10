"""
tf_target_edge -- Mouse TF→target relationships from DoRothEA and TFLink.

Returns a signed regulatory weight (positive = activation, negative = repression)
and confidence level from two curated databases:
  - DoRothEA mm: literature-curated TF regulons, confidence A-E
  - TFLink Mus musculus: large-scale TF-target interactions

Data loaded from local files once and cached in-process.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import os as _os
_grn_env = _os.environ.get("GRN_DATA_DIR")
GRN = Path(_grn_env) if _grn_env else Path(__file__).resolve().parents[5] / "GRN Inference" / "GRN-Embeddings" / "data"
DOROTHEA_PATH = GRN / "DoRothEA" / "dorothea_mm.csv"
TFLINK_PATH   = GRN / "TFLink" / "TFLink_Mus_musculus_interactions_All_simpleFormat_v1.0.tsv.gz"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "tf_target_edge",
        "description": (
            "Look up whether a perturbation gene directly regulates a target gene "
            "in mouse, using DoRothEA and TFLink databases. Returns the regulatory "
            "weight (positive=activation, negative=repression), confidence level, "
            "and source database. Returns 'no direct TF-target edge found' if neither "
            "database contains the pair."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pert":   {"type": "string", "description": "Perturbation gene symbol (e.g. 'Stat1')."},
                "target": {"type": "string", "description": "Target gene symbol (e.g. 'Irf1')."},
            },
            "required": ["pert", "target"],
        },
    },
}

_CONFIDENCE_ORDER = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


@lru_cache(maxsize=1)
def _load_dorothea() -> dict[tuple[str, str], dict]:
    db: dict[tuple[str, str], dict] = {}
    if not DOROTHEA_PATH.exists():
        return db
    with open(DOROTHEA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["source"].lower(), row["target"].lower())
            existing = db.get(key)
            # keep highest-confidence entry
            if existing is None or _CONFIDENCE_ORDER.get(row["confidence"], 0) > \
               _CONFIDENCE_ORDER.get(existing["confidence"], 0):
                db[key] = {
                    "weight": float(row["weight"]),
                    "confidence": row["confidence"],
                }
    return db


@lru_cache(maxsize=1)
def _load_tflink() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if not TFLINK_PATH.exists():
        return pairs
    import gzip
    with gzip.open(TFLINK_PATH, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tf  = row.get("Name.TF", "").lower()
            tgt = row.get("Name.Target", "").lower()
            if tf and tgt:
                pairs.add((tf, tgt))
    return pairs


def tf_target_edge(pert: str, target: str) -> str:
    """Return TF→target regulatory edge info from DoRothEA and TFLink."""
    key = (pert.lower(), target.lower())
    lines = []

    dorothea = _load_dorothea()
    if key in dorothea:
        e = dorothea[key]
        direction = "activates" if e["weight"] > 0 else "represses"
        lines.append(
            f"DoRothEA: {pert} {direction} {target} "
            f"(weight={e['weight']:+.2f}, confidence={e['confidence']})"
        )

    tflink = _load_tflink()
    if key in tflink:
        lines.append(f"TFLink: {pert} → {target} interaction confirmed in Mus musculus.")

    if not lines:
        return f"No direct TF-target edge found for {pert} → {target} in DoRothEA or TFLink."

    return "\n".join(lines)
