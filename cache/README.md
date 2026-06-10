# cache/

Pre-built and auto-populated lookup caches. These files are read at inference time to avoid redundant API calls.

## Files

| File | Source | Gitignored? |
|------|--------|-------------|
| `replogle_k562_signatures.json` | Built by `scripts/build_replogle_cache.py` from Replogle et al. 2022 | No — committed (894 KB, needed at runtime) |
| `mygene_cache.json` | Populated at runtime by `tools/gene_info.py` | Yes |
| `string_cache.json` | Populated at runtime by `tools/protein_interactions.py` | Yes |

The API caches (`mygene_cache.json`, `string_cache.json`) grow as the agent queries new genes. Pre-warm them before a LUMI run with:

```bash
uv run python scripts/warm_cache.py
```
