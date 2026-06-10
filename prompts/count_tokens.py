"""
Count tokens for every prompt in prompts/.

Uses tiktoken with o200k_base encoding (GPT-4o / o-series models).
Reports token count, character count, and whether each prompt fits within
the Track B system prompt limit of 4,096 tokens.

Usage:
    uv run python prompts/count_tokens.py
    uv run python prompts/count_tokens.py --limit 4096
"""
from __future__ import annotations

import argparse
from pathlib import Path

TRACK_B_LIMIT = 4096
ENCODING = "o200k_base"  # GPT-4o / o-series (GPT-OSS-120B) token vocabulary


def count_tokens(text: str) -> int:
    import tiktoken
    enc = tiktoken.get_encoding(ENCODING)
    return len(enc.encode(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Count tokens in prompt files")
    parser.add_argument(
        "--limit", type=int, default=TRACK_B_LIMIT,
        help=f"Token budget to check against (default: {TRACK_B_LIMIT} for Track B)",
    )
    parser.add_argument(
        "--dir", type=Path, default=Path(__file__).parent,
        help="Directory to scan for .txt prompt files",
    )
    args = parser.parse_args()

    prompts = sorted(args.dir.glob("*.txt"))
    if not prompts:
        print(f"No .txt files found in {args.dir}")
        return

    print(f"Encoding : {ENCODING}")
    print(f"Limit    : {args.limit} tokens")
    print()
    print(f"{'File':<35} {'Tokens':>7}  {'Chars':>7}  {'Status'}")
    print("-" * 65)

    for path in prompts:
        text = path.read_text()
        tokens = count_tokens(text)
        chars = len(text)
        remaining = args.limit - tokens
        if remaining >= 0:
            status = f"OK  ({remaining} tokens remaining)"
        else:
            status = f"OVER LIMIT by {-remaining} tokens"
        print(f"{path.name:<35} {tokens:>7,}  {chars:>7,}  {status}")


if __name__ == "__main__":
    main()
