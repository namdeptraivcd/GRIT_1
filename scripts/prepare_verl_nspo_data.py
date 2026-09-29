#!/usr/bin/env python3
"""Convert GRIT task parquet rows to the conversational format expected by verl.

The NSPO/verl dataset loader expects ``prompt`` to be a list of chat messages.
The standalone GRIT trainer uses a formatted string instead, so this adapter
keeps the source task data unchanged and writes a small RLHF-compatible copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source parquet with raw_prompt/prompt columns.")
    parser.add_argument("--output", required=True, help="Output parquet for verl RLHFDataset.")
    parser.add_argument("--prompt-column", default="raw_prompt")
    parser.add_argument("--limit", type=int, default=0, help="Optional deterministic prefix limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import Dataset, load_dataset

    source = load_dataset("parquet", data_files=args.input, split="train")
    if args.prompt_column not in source.column_names:
        raise ValueError(
            f"Column {args.prompt_column!r} not found in {args.input}; "
            f"available={source.column_names}"
        )
    if args.limit > 0:
        source = source.select(range(min(args.limit, len(source))))

    rows = []
    for index, row in enumerate(source):
        text = str(row[args.prompt_column]).strip()
        if not text:
            continue
        rows.append(
            {
                "prompt": [{"role": "user", "content": text}],
                "data_source": "PKU-SafeRLHF",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"index": index, "raw_prompt": text},
            }
        )
    if not rows:
        raise ValueError("No non-empty prompts were found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))
    print(f"wrote {len(rows)} verl prompts to {output}")


if __name__ == "__main__":
    main()
