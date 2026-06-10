# prompts/

System prompts for the agent. `track_b_submit.py` reads `prompt.txt` by default; pass `--system-prompt prompts/other.txt` to try a different one.

Track B allows up to **4,096 prompt tokens**. Check any prompt with:

```bash
uv run python prompts/count_tokens.py
```

## Files

| File | Tokens | Description |
|------|--------|-------------|
| `prompt.txt` | 633 | Expert macrophage immunologist — 6-step reasoning strategy with tool budget guidance |

## Adding a new prompt

1. Save it as `prompts/<name>.txt`
2. Run `count_tokens.py` to verify it fits within 4,096 tokens
3. Test it: `uv run python track_b_submit.py --system-prompt prompts/<name>.txt --eval-train --rows 50 --clear-cache`
