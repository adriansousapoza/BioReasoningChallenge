# tools/

The six retrieval tools available to the agent during Track B inference. Each tool is a Python function with a `TOOL_SCHEMA` dict (OpenAI function-calling format) that DSPy uses to build the tool list.

The agent has a budget of **12 tool calls per question**. Recommended order: start with `tf_target_edge` + `perturb_seq_lookup` (highest signal), then `gene_info` / `protein_interactions` if still uncertain, and `base_rates_and_examples` to anchor final probabilities.

## Tools

| File | Tool name | Source | What it returns |
|------|-----------|--------|-----------------|
| `gene_info.py` | `gene_info` | [mygene.info](https://mygene.info) API | Gene summary, type (TF/kinase/etc.), GO biological process terms, KEGG pathways |
| `protein_interactions.py` | `protein_interactions` | [STRING DB](https://string-db.org) API | Top PPI partners with combined confidence scores |
| `train_data_lookup.py` | `train_data_lookup` | Local `data/train.csv` | Known up/down/none labels for pairs involving the same pert or gene |
| `tf_target_edge.py` | `tf_target_edge` | Local GRN (`$BIOREASONDATA/grn/`) | Direct TF→target regulatory edges with evidence source and direction |
| `perturb_seq_lookup.py` | `perturb_seq_lookup` | Replogle et al. 2022 K562 Perturb-seq | Whether knocking down the human ortholog changes target expression in K562 cells |
| `base_rates_and_examples.py` | `base_rates_and_examples` | Local `data/train.csv` | Global label priors (P(none)≈0.55, P(up\|DE)≈0.69) + pert-specific training examples |

## Caching

`gene_info` and `protein_interactions` cache API responses to `cache/mygene_cache.json` and `cache/string_cache.json` respectively. Run `scripts/warm_cache.py` before a LUMI job to pre-populate the caches and avoid API calls during inference.

`perturb_seq_lookup` reads from `cache/replogle_k562_signatures.json` (pre-built, committed to the repo). Rebuild it with `scripts/build_replogle_cache.py` if needed.

## Adding a new tool

1. Create `tools/my_tool.py` with a function `my_tool(...)` and a `TOOL_SCHEMA` dict.
2. Import and add it to `tool_list` in `track_b_submit.py`.
3. Document it here.
