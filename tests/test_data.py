import pytest

from grit1.data import distributed_batches, extract_prompt, validate_dataset_role


def test_distributed_batches_have_equal_step_counts():
    rows = [{"prompt": str(index)} for index in range(11)]
    rank_batches = [
        list(
            distributed_batches(
                rows,
                batch_size=2,
                rank=rank,
                world_size=3,
                seed=66,
                epoch=0,
                drop_last=True,
            )
        )
        for rank in range(3)
    ]
    assert [len(batches) for batches in rank_batches] == [1, 1, 1]
    assert all(len(batch) == 2 for batches in rank_batches for batch in batches)


def test_extract_prompt_prefers_raw_prompt():
    assert extract_prompt({"prompt": "formatted", "raw_prompt": "raw"}) == "raw"


def test_dataset_roles_prevent_task_preservation_mixups():
    validate_dataset_role(
        [{"dataset_role": "safety_task", "prompt": "task"}],
        expected="safety_task",
        source="task.parquet",
    )
    validate_dataset_role(
        [{"prompt": "legacy external row"}],
        expected="preservation",
        source="external.parquet",
    )
    with pytest.raises(ValueError, match="dataset_role"):
        validate_dataset_role(
            [{"dataset_role": "safety_task", "prompt": "wrong input"}],
            expected="preservation",
            source="preservation.parquet",
        )
