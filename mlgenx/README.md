# mlgenx/

The official challenge helper package from [Genentech/BioReasoningChallenge](https://github.com/genentech/bioreasoningchallenge). Installed as an editable package via `uv sync`.

## Modules

### `prompts.py`

Generates per-question prompts (zero-shot or few-shot) for `(pert, gene)` pairs. The zero-shot template is used as the **user message** in each agentic turn; the **system message** comes from `prompts/prompt.txt`.

Key exports:
- `format_prompt(pert, gene, examples=None)` — single prompt string
- `format_prompts_from_csv(csv_path)` — DataFrame of `{id, prompt}` for a full CSV
- `CELL_DESC` — cell line context string injected into prompts

### `parsing.py`

Parses LLM free-text responses into `(prediction_up, prediction_down)` float pairs.

Key exports:
- `parse_answer(text)` — returns `(float, float)` from A/B/C letter or free text
- `parse_answers(texts)` — batch version

### `__init__.py`

Re-exports `format_prompt`, `format_prompts_from_csv`, `parse_answer`, `parse_answers`, `build_submission` for convenience.
