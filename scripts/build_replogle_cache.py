"""
Build the Replogle K562 Essential Perturb-seq signature cache from Harmonizome.

Downloads perturbation signatures for all pert genes' human orthologs and saves
them to replogle_k562_signatures.json (one entry per mouse pert gene).

Run once locally before transferring to LUMI:
    python scripts/build_replogle_cache.py
"""
import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "replogle_k562_signatures.json"
DATASET    = "Replogle+et+al.%2C+Cell%2C+2022+K562+Essential+Perturb-seq+Gene+Perturbation+Signatures"
BASE       = "https://maayanlab.cloud/Harmonizome/api/1.0"

train = pd.read_csv(ROOT / "data/train.csv")
test  = pd.read_csv(ROOT / "data/test.csv")
PERTS = sorted(set(train["pert"]) | set(test["pert"]))


def harmonizome_get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.load(r)


def get_signature(attr_name: str) -> dict:
    path = f"/gene_set/{urllib.parse.quote(attr_name)}/{DATASET}"
    d = harmonizome_get(path)
    return {a["gene"]["symbol"]: a.get("standardizedValue", 0.0)
            for a in d.get("associations", [])}


# Load dataset index
print("Loading Harmonizome dataset index...")
ds       = harmonizome_get(f"/dataset/{DATASET}")
gene_sets = ds.get("geneSets", [])
print(f"Gene sets available: {len(gene_sets)}")

sym_to_attr: dict[str, str] = {}
for gs in gene_sets:
    attr = gs["name"].split("/")[0]
    m = re.match(r"\d+_([A-Z0-9]+)_", attr)
    if m:
        sym_to_attr[m.group(1)] = attr

covered = [p for p in PERTS if p.upper() in sym_to_attr]
print(f"Pert genes with K562 signature: {len(covered)}/{len(PERTS)}")

# Load existing cache
cache: dict = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
to_fetch = [p for p in covered if p not in cache]
print(f"Fetching {len(to_fetch)} new signatures...")

for i, pert in enumerate(to_fetch):
    attr = sym_to_attr[pert.upper()]
    try:
        cache[pert] = get_signature(attr)
    except Exception as e:
        print(f"  Error {pert}: {e}")
    if i % 20 == 0:
        CACHE_PATH.write_text(json.dumps(cache))
        print(f"  {i+1}/{len(to_fetch)} ({pert})")
    time.sleep(0.1)

CACHE_PATH.write_text(json.dumps(cache))
print(f"\nDone. {len(cache)} signatures saved to {CACHE_PATH}")
