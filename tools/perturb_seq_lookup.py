"""
perturb_seq_lookup -- K562 CRISPRi Perturb-seq signatures via human ortholog.

Looks up whether a target gene is differentially expressed in human K562 cells
when the perturbation gene's human ortholog is knocked down (Replogle et al.,
Cell 2022, essential-gene subset; downloaded via Harmonizome).

Transfer confidence is HIGH for conserved essential machinery (ribosome,
proteasome, splicing, CCT, ER) and LOWER for immune/inflammatory genes
(K562 has no LPS/cytokine context).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import os as _os
_data_env = _os.environ.get("BIOREASONDATA")
CACHE_PATH = (Path(_data_env) / "replogle_k562_signatures.json") if _data_env else \
             Path(__file__).resolve().parents[1] / "cache" / "replogle_k562_signatures.json"

# Gene categories with weak K562→BMDM transfer
_IMMUNE_KEYWORDS = {
    "jak", "stat", "irf", "nfkb", "tlr", "traf", "irak", "tyk",
    "ifn", "il", "tnf", "syk", "trem", "tyrobp", "cd", "mhc",
}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "perturb_seq_lookup",
        "description": (
            "Look up how knocking down a gene affects a target gene's expression "
            "in human K562 cells (Replogle 2022 Essential Perturb-seq). "
            "Returns the standardized effect score and transfer confidence "
            "(HIGH for conserved machinery, LOW for immune/inflammatory genes "
            "that differ between K562 and mouse macrophages)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pert":   {"type": "string", "description": "Perturbation gene symbol."},
                "target": {"type": "string", "description": "Target gene symbol."},
            },
            "required": ["pert", "target"],
        },
    },
}


@lru_cache(maxsize=1)
def _load_sigs() -> dict:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH) as f:
        return json.load(f)


def _transfer_confidence(pert: str) -> str:
    p = pert.lower()
    for kw in _IMMUNE_KEYWORDS:
        if kw in p:
            return "LOW (immune gene — K562 lacks macrophage signalling context)"
    return "HIGH (conserved essential machinery)"


def perturb_seq_lookup(pert: str, target: str) -> str:
    """Return K562 Perturb-seq score for pert→target, with transfer note."""
    sigs = _load_sigs()
    if pert not in sigs:
        # Try case variants
        for k in sigs:
            if k.lower() == pert.lower():
                pert = k
                break
        else:
            return (
                f"No K562 Perturb-seq signature found for '{pert}'. "
                f"This gene may not be essential in K562 cells or was not profiled."
            )

    sig = sigs[pert]
    target_human = target.upper()
    score = sig.get(target_human)

    conf = _transfer_confidence(pert)

    # Always show top hits for context
    top = sorted(sig.items(), key=lambda x: -abs(x[1]))[:8]
    top_str = ", ".join(f"{g}:{s:+.2f}" for g, s in top)

    if score is None:
        return (
            f"K562 Perturb-seq ({pert} knockdown): {target} was NOT significantly affected.\n"
            f"Transfer confidence: {conf}\n"
            f"Top affected genes in K562: {top_str}"
        )

    direction = "UP-regulated" if score > 0 else "DOWN-regulated"
    strength  = "strongly" if abs(score) >= 1.5 else ("moderately" if abs(score) >= 0.5 else "weakly")
    return (
        f"K562 Perturb-seq ({pert} knockdown): {target} is {strength} {direction} "
        f"(standardized score={score:+.3f}).\n"
        f"Transfer confidence: {conf}\n"
        f"Top affected genes in K562: {top_str}"
    )
