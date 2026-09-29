#!/usr/bin/env python3
"""Sample NSPO-domain prompts, then generate fixed-base GRIT contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.preservation_data import preservation_prompt_ids, sample_source, tokenizer_fingerprint
except ModuleNotFoundError as exc:
    # When this file is invoked as `python scripts/prepare_preservation_data.py`,
    # Python puts `scripts/` (not the repository root) first on sys.path.
    # Fall back to the sibling module while preserving unrelated import errors.
    if exc.name != "scripts.preservation_data":
        raise
    from preservation_data import preservation_prompt_ids, sample_source, tokenizer_fingerprint


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        return json.load(stream)


def finish_output(output: Path, rows: list[dict], manifest: dict, filename: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = output / filename
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    manifest.update({
        "rows": len(rows),
        "domain_counts": dict(Counter(row["domain"] for row in rows)),
        "parquet": filename,
        "parquet_sha256": file_hash(parquet),
    })
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def sample(args) -> None:
    from huggingface_hub import hf_hub_download

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    seen, rows, sources = set(), [], []
    for source in config["sources"]:
        print(f"Loading {source['repo_id']} / {source['split']}", flush=True)
        path = Path(hf_hub_download(
            repo_id=source["repo_id"], filename=source["filename"],
            revision=source["revision"], repo_type="dataset", cache_dir=args.cache_dir,
        ))
        selected = sample_source(read_rows(path), source, config["seed"], seen)
        rows.extend(selected)
        sources.append({**source, "file_sha256": file_hash(path)})
    random.Random(config["seed"]).shuffle(rows)
    (output / "prompts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    finish_output(output, rows, {
        "dataset_role": "preservation",
        "stage": "prompts_only", "seed": config["seed"], "sources": sources,
        "sampling": "per-domain shuffled rows, global normalized prompt deduplication",
        "nspo_note": "Source families match NSPO citations; splits and ratios are GRIT choices.",
    }, "preserve_prompts.parquet")


def generate(args) -> None:
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if args.max_prompt_length < 1 or args.max_new_tokens < 1:
        raise ValueError("Token limits must be positive")
    rows = read_rows(args.prompts)
    if not rows:
        raise ValueError("No preservation prompts")
    if args.limit:
        rows = rows[:args.limit]
    revision = model_info(args.model_path, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    fingerprint = tokenizer_fingerprint(tokenizer)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, revision=revision, torch_dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    model.requires_grad_(False)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    generated = []
    with (output / "contexts.partial.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            prompt_ids = preservation_prompt_ids(tokenizer, row["prompt"])
            # Do not truncate the problem or the chat template silently.
            if len(prompt_ids) > args.max_prompt_length:
                raise ValueError(f"{row['id']}: prompt exceeds --max-prompt-length")
            set_seed(args.seed + index)
            inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                sequence = model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )[0].tolist()
            response_ids = sequence[len(prompt_ids):]
            if not response_ids:
                raise ValueError(f"{row['id']}: empty generated response")
            record = {
                **row,
                "text": tokenizer.decode(sequence, skip_special_tokens=False),
                "base_response": tokenizer.decode(response_ids, skip_special_tokens=True),
                "input_ids": sequence,
                "response_start": len(prompt_ids),
                "response_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
                "tokenizer_sha256": fingerprint,
                "base_model": args.model_path,
                "base_revision": revision,
            }
            generated.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"Generated {index + 1}/{len(rows)}", flush=True)
    finish_output(output, generated, {
        "dataset_role": "preservation",
        "stage": "base_contexts", "base_model": args.model_path, "base_revision": revision,
        "tokenizer_sha256": fingerprint, "seed": args.seed, "do_sample": False,
        "max_prompt_length": args.max_prompt_length, "max_new_tokens": args.max_new_tokens,
        "input_sha256": file_hash(args.prompts),
    }, "preserve_contexts.parquet")
    (output / "contexts.partial.jsonl").rename(output / "contexts.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    sampling = commands.add_parser("sample")
    sampling.add_argument("--config", type=Path, default=REPO_ROOT / "config/preservation/nspo_mix.json")
    sampling.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/preservation/nspo_mix")
    sampling.add_argument("--cache-dir", default=str(REPO_ROOT / ".cache/huggingface"))
    generation = commands.add_parser("generate")
    generation.add_argument("--prompts", type=Path, required=True)
    generation.add_argument("--output-dir", type=Path, required=True)
    generation.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    generation.add_argument("--revision", default="main")
    generation.add_argument("--seed", type=int, default=66)
    generation.add_argument("--max-prompt-length", type=int, default=2048)
    generation.add_argument("--max-new-tokens", type=int, default=256)
    generation.add_argument("--device", default=None)
    generation.add_argument("--limit", type=int, default=0, help="Positive count for generation smoke tests")
    args = parser.parse_args()
    if args.stage == "generate" and args.limit < 0:
        parser.error("--limit must be non-negative")
    (sample if args.stage == "sample" else generate)(args)


if __name__ == "__main__":
    main()
