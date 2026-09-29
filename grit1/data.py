from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator, Sequence


DATASET_ROLES = {"safety_task", "preservation"}


def load_rows(path_or_dataset: str, *, split: str = "train") -> list[dict[str, Any]]:
    path = Path(path_or_dataset)
    if path.suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
        return rows
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{path} must contain a list of JSON objects")
        return rows

    from datasets import load_dataset

    if path.suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(path), split=split)
    elif path.suffix == ".csv":
        dataset = load_dataset("csv", data_files=str(path), split=split)
    else:
        dataset = load_dataset(path_or_dataset, split=split)
    return [dict(row) for row in dataset]


def extract_prompt(row: dict[str, Any]) -> str:
    for key in ("raw_prompt", "prompt", "raw_text", "text", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"row has no usable prompt field: {sorted(row)}")


def validate_dataset_role(
    rows: Sequence[dict[str, Any]], *, expected: str, source: str
) -> None:
    """Reject generated datasets wired into the wrong training input.

    Legacy or external datasets without ``dataset_role`` remain supported.
    Once a role marker is present, every marked row must match the expected
    input so safety-task examples cannot silently become preservation data.
    """

    if expected not in DATASET_ROLES:
        raise ValueError(f"unknown expected dataset role: {expected!r}")
    observed = {
        str(row["dataset_role"])
        for row in rows
        if row.get("dataset_role") not in (None, "")
    }
    unexpected = observed - {expected}
    if unexpected:
        raise ValueError(
            f"{source} is wired as {expected!r}, but contains dataset_role="
            f"{sorted(unexpected)!r}"
        )


def distributed_batches(
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    epoch: int,
    drop_last: bool = False,
) -> Iterator[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")

    indices = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(indices)
    if drop_last:
        global_batch = batch_size * world_size
        indices = indices[: len(indices) - (len(indices) % global_batch)]
    indices = indices[rank::world_size]
    for start in range(0, len(indices), batch_size):
        batch = [rows[index] for index in indices[start : start + batch_size]]
        if len(batch) < batch_size and drop_last:
            continue
        if batch:
            yield batch
