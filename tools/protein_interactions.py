"""
protein_interactions -- Fetch protein-protein interactions from STRING DB.

Queries the STRING DB REST API for known interaction partners of a mouse
protein, returning partner names and combined confidence scores.

API docs: https://string-db.org/help/api/
"""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import os as _os
_data_env = _os.environ.get("BIOREASONDATA")
_CACHE_PATH = (Path(_data_env) / "cache" / "string_cache.json") if _data_env else \
              Path(__file__).resolve().parents[2] / "cache" / "string_cache.json"


@lru_cache(maxsize=1)
def _offline_cache() -> dict:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "protein_interactions",
        "description": (
            "Fetch known protein-protein interactions for a mouse gene "
            "from STRING DB.  Returns up to 10 interaction partners "
            "with combined confidence scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gene_symbol": {
                    "type": "string",
                    "description": "Mouse gene symbol (e.g. 'Stat1').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max interaction partners to return (default 10).",
                },
            },
            "required": ["gene_symbol"],
        },
    },
}

_API_URL = (
    "https://string-db.org/api/json/interaction_partners?"
    "identifiers={symbol}&species=10090&limit={limit}"
)


def protein_interactions(gene_symbol: str, limit: int = 10) -> str:
    """Fetch PPI partners and return a human-readable summary."""
    limit = min(max(1, limit), 50)
    # Check offline cache first (used on LUMI compute nodes without internet)
    cached = _offline_cache().get(gene_symbol)
    if cached is not None:
        data = cached
    else:
        url = _API_URL.format(symbol=gene_symbol, limit=limit)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            return f"Error querying STRING DB for {gene_symbol}: {e}"

    if not data:
        return (
            f"No protein interactions found for '{gene_symbol}' "
            f"in mouse (STRING DB)."
        )

    lines = [f"Protein interactions for {gene_symbol} (mouse, STRING DB):"]
    for entry in data[:limit]:
        partner = entry.get("preferredName_B", entry.get("stringId_B", "?"))
        score = entry.get("score", 0)
        lines.append(f"  - {partner} (combined score: {score:.3f})")

    return "\n".join(lines)
