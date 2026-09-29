import argparse
import importlib.util
from collections import Counter
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "prepare_grit_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_grit_data", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_grit_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_grit_data)


def test_output_directories_are_physically_separate(tmp_path):
    args = argparse.Namespace(
        data_root=str(tmp_path / "grit1"),
        task_output_dir=None,
        preservation_output_dir=None,
    )
    task_dir, preservation_dir = prepare_grit_data.resolve_output_dirs(args)
    assert task_dir == tmp_path / "grit1" / "safety_task"
    assert preservation_dir == tmp_path / "grit1" / "preservation"

    args.preservation_output_dir = str(task_dir)
    with pytest.raises(ValueError, match="must be different"):
        prepare_grit_data.resolve_output_dirs(args)


def test_task_and_preservation_rows_have_distinct_roles():
    task = prepare_grit_data.row_to_task(
        {
            "prompt": "How should I stay safe?",
            "response_0": "Safe response",
            "response_1": "Unsafe response",
            "safer_response_id": 0,
        },
        prefer_safe=True,
    )
    preservation = prepare_grit_data.build_nspo_preserve_rows(
        [
            ("common_sense", "general-fixture", [{"instruction": "Explain", "input": "gravity"}]),
            ("code", "code-fixture", [{"query": "Write a loop"}]),
            ("math", "math-fixture", [{"question": "What is 2+2?"}]),
        ],
        max_samples=3,
        seed=66,
    )

    assert task["dataset_role"] == "safety_task"
    assert all(row["dataset_role"] == "preservation" for row in preservation)
    assert Counter(row["domain"] for row in preservation) == {
        "common_sense": 1,
        "code": 1,
        "math": 1,
    }
    assert all("chosen" not in row and "rejected" not in row for row in preservation)


def test_default_nspo_quota_matches_grit_without_copying_answers():
    rows = prepare_grit_data.build_nspo_preserve_rows(
        [
            (
                "common_sense",
                "tatsu-lab/alpaca_farm",
                [
                    {
                        "instruction": f"instruction-{index}",
                        "input": f"context-{index}",
                        "output": "unused-answer",
                    }
                    for index in range(400)
                ],
            ),
            (
                "code",
                "newfacade/LeetCodeDataset",
                [
                    {"query": f"code-{index}", "completion": "unused-answer"}
                    for index in range(400)
                ],
            ),
            (
                "math",
                "openai/gsm8k",
                [
                    {"question": f"math-{index}", "answer": "unused-answer"}
                    for index in range(400)
                ],
            ),
        ],
        max_samples=1000,
        seed=66,
    )

    assert Counter(row["domain"] for row in rows) == {
        "common_sense": 334,
        "code": 333,
        "math": 333,
    }
    assert not any("unused-answer" in row["raw_text"] for row in rows)
