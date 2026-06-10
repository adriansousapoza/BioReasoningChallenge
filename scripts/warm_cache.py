"""
Pre-cache all internet-dependent tool calls for offline use on LUMI.

Run this locally (where internet is available) before transferring to LUMI:
    python scripts/warm_cache.py

Populates:
    cache/mygene_cache.json   — mygene.info annotations for all genes
    cache/string_cache.json   — STRING DB interactions for all pert genes
"""
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

TRAIN = pd.read_csv(ROOT / "data/train.csv")
TEST  = pd.read_csv(ROOT / "data/test.csv")
ALL_GENES = sorted(set(TRAIN["pert"]) | set(TEST["pert"]) |
                   set(TRAIN["gene"]) | set(TEST["gene"]))
PERT_GENES = sorted(set(TRAIN["pert"]) | set(TEST["pert"]))

print(f"Genes to cache: {len(ALL_GENES)} total, {len(PERT_GENES)} pert genes")


# ---------------------------------------------------------------------------
# mygene.io
# ---------------------------------------------------------------------------

def warm_mygene():
    path = CACHE_DIR / "mygene_cache.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo  = [g for g in ALL_GENES if g not in cache]
    print(f"\n[mygene.io] {len(cache)} cached, {len(todo)} to fetch...")

    for i, g in enumerate(todo):
        try:
            url = (f"https://mygene.info/v3/query?"
                   f"q=symbol:{urllib.parse.quote(g)}&species=mouse"
                   f"&fields=symbol,name,summary,go.BP,pathway.kegg&size=1")
            with urllib.request.urlopen(url, timeout=10) as r:
                cache[g] = json.load(r)
        except Exception as e:
            cache[g] = {"hits": [], "error": str(e)}

        if i % 100 == 0:
            path.write_text(json.dumps(cache))
            print(f"  mygene: {i+1}/{len(todo)}")
        time.sleep(0.05)   # polite rate limit

    path.write_text(json.dumps(cache))
    print(f"[mygene.io] Done — {len(cache)} genes cached at {path}")


# ---------------------------------------------------------------------------
# STRING DB (only pert genes — reduces requests from ~2200 to ~482)
# ---------------------------------------------------------------------------

def warm_string():
    path  = CACHE_DIR / "string_cache.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo  = [g for g in PERT_GENES if g not in cache]
    print(f"\n[STRING DB] {len(cache)} cached, {len(todo)} to fetch...")

    for i, g in enumerate(todo):
        try:
            url = (f"https://string-db.org/api/json/interaction_partners?"
                   f"identifiers={urllib.parse.quote(g)}&species=10090&limit=15")
            with urllib.request.urlopen(url, timeout=15) as r:
                cache[g] = json.load(r)
        except Exception as e:
            cache[g] = []
            print(f"  STRING error {g}: {e}")

        if i % 50 == 0:
            path.write_text(json.dumps(cache))
            print(f"  string: {i+1}/{len(todo)}")
        time.sleep(0.2)   # STRING requests slower rate limit

    path.write_text(json.dumps(cache))
    print(f"[STRING DB] Done — {len(cache)} genes cached at {path}")


if __name__ == "__main__":
    warm_mygene()
    warm_string()
    print("\nAll caches ready. Transfer with:")
    print(f"  rsync -av {CACHE_DIR}/ sousapoz@lumi.csc.fi:/scratch/project_XXXXXXX/sousapoz/code/cache/")
