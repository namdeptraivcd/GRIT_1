"""Source formatting and exact-token preservation batches."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata


def prompt_key(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def source_prompt(row: dict, domain: str) -> str:
    if domain == "general":
        return "\n\n".join(
            value.strip() for value in (row["instruction"], row.get("input", ""))
            if value and value.strip()
        )
    field = {"math": "question", "code": "query"}[domain]
    return (row[field] or "").strip()


def sample_source(rows, source: dict, seed: int, seen: set[str]) -> list[dict]:
    count = source["count"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Each source count must be a positive integer")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = []
    for index in indices:
        row = rows[index]
        prompt = source_prompt(row, source["domain"])
        key = prompt_key(prompt)
        if not prompt or key in seen:
            continue
        seen.add(key)
        selected.append({
            "dataset_role": "preservation",
            "id": f"{source['repo_id']}:{source['revision']}:{source['split']}:{index}",
            "source_id": str(row.get("task_id", index)),
            "source_row": index,
            "data_source": source["repo_id"],
            "source_revision": source["revision"],
            "source_split": source["split"],
            "domain": source["domain"],
            "prompt": prompt,
            "text": prompt,
            "prompt_sha256": key,
        })
        if len(selected) == count:
            return selected
    raise ValueError(f"{source['repo_id']}: need {count} unique prompts, found {len(selected)}")


def tokenizer_fingerprint(tokenizer) -> str:
    payload = {
        "vocab": tokenizer.get_vocab(),
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": tokenizer.chat_template,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def preservation_prompt_ids(tokenizer, prompt: str) -> list[int]:
    """Request an explicit encoding schema across Transformers 4/5."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_dict=True,
    )
    return list(encoded["input_ids"])


def stored_context_arrays(tokenizer, rows, *, max_length: int):
    """Pad exact contexts on the right; retain the original response boundary."""
    fingerprint = getattr(tokenizer, "_grit_preservation_fingerprint", None)
    if fingerprint is None:
        fingerprint = tokenizer_fingerprint(tokenizer)
        tokenizer._grit_preservation_fingerprint = fingerprint
    sequences, masks = [], []
    for row in rows:
        if row["tokenizer_sha256"] != fingerprint:
            raise ValueError("Preservation tokenizer differs from the training tokenizer")
        full_ids = list(row["input_ids"])
        start = int(row["response_start"])
        if not 1 <= start < len(full_ids):
            raise ValueError("Invalid preservation response_start")
        ids = full_ids[:max_length]
        if start >= len(ids):
            raise ValueError("max_preserve_length removes all response tokens; increase it")
        sequences.append(ids)
        masks.append([0] * start + [1] * (len(ids) - start))
    width = max(map(len, sequences))
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Preservation batching requires a pad token")
    input_ids = [ids + [pad] * (width - len(ids)) for ids in sequences]
    attention = [[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences]
    response = [mask + [0] * (width - len(mask)) for mask in masks]
    return input_ids, attention, response


def stored_preservation_batch(tokenizer, rows, *, max_length: int):
    """Return next-token labels and response-only mask without retokenization."""
    import torch

    ids, attention, response = stored_context_arrays(tokenizer, rows, max_length=max_length)
    input_ids = torch.tensor(ids, dtype=torch.long)
    attention = torch.tensor(attention, dtype=torch.long)
    response = torch.tensor(response, dtype=torch.bool)
    return input_ids, attention, input_ids[:, 1:].contiguous(), response[:, 1:].contiguous()
