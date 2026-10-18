"""MBPP split loading and deterministic evaluation-task selection.

This module is deliberately the only place that decides which MBPP records are
training data and which records are held out.  Prompt construction is added in
the next task; keeping the split logic here makes it hard for later scripts to
accidentally train on evaluation problems.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Mapping, Sequence


# ``mbpp`` was the old datasets-library alias.  The fully qualified repository
# works with current datasets releases as well.
MBPP_DATASET = "google-research-datasets/mbpp"
MBPP_CONFIG = "full"
TRAIN_SPLITS = ("train", "validation", "prompt")
TEST_SPLIT = "test"
EVAL_TASK_COUNT = 100
EVAL_SEED = 0


@dataclass(frozen=True)
class MBPPSplits:
    """The fixed MBPP partitions used by RoleKV.

    ``training`` combines the three allowed source splits. ``evaluation`` is
    the deterministic 100-task subset of the official MBPP test split.
    """

    training: Any
    evaluation: Any


def _task_ids(rows: Sequence[Mapping[str, Any]]) -> set[Any]:
    """Return task IDs and fail early if a dataset unexpectedly lacks them."""

    try:
        ids = [row["task_id"] for row in rows]
    except KeyError as exc:
        raise ValueError("MBPP records must contain a 'task_id' field") from exc

    if len(ids) != len(set(ids)):
        raise ValueError("MBPP split contains duplicate task_id values")
    return set(ids)


def select_evaluation_tasks(
    test_split: Any,
    *,
    count: int = EVAL_TASK_COUNT,
    seed: int = EVAL_SEED,
) -> Any:
    """Select a reproducible subset of an MBPP test split.

    Sampling uses row positions from the published test split, so all machines
    choose exactly the same records when they use the same dataset revision.
    """

    if count < 1:
        raise ValueError("count must be at least 1")
    if len(test_split) < count:
        raise ValueError(
            f"requested {count} evaluation tasks, but test split has only {len(test_split)}"
        )

    _task_ids(test_split)
    indices = random.Random(seed).sample(range(len(test_split)), count)
    return test_split.select(indices)


def build_utils_py(
    task: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    *,
    ctx_funcs: int = 30,
) -> str:
    """Build a deterministic ``utils.py`` from other MBPP reference solutions.

    The held-out task is always excluded.  The selected functions are sorted by
    task ID after sampling, which makes the rendered context stable even if the
    input rows come from an unordered collection.
    """

    if ctx_funcs < 0:
        raise ValueError("ctx_funcs must be non-negative")
    if "task_id" not in task:
        raise ValueError("task must contain a 'task_id' field")

    task_id = task["task_id"]
    candidates = [
        row
        for row in training_rows
        if row.get("task_id") != task_id and isinstance(row.get("code"), str) and row["code"].strip()
    ]
    candidates.sort(key=lambda row: str(row["task_id"]))
    if ctx_funcs > len(candidates):
        raise ValueError(f"requested {ctx_funcs} context functions, but only {len(candidates)} are available")

    seed_bytes = hashlib.sha256(f"rolekv-utils-v1:{task_id}".encode()).digest()
    seed = int.from_bytes(seed_bytes[:8], byteorder="big")
    selected = random.Random(seed).sample(candidates, ctx_funcs)
    selected.sort(key=lambda row: str(row["task_id"]))
    return "\n\n".join(row["code"].strip() for row in selected) + ("\n" if selected else "")


def build_task_segment(
    task: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    *,
    ctx_funcs: int = 30,
) -> str:
    """Render one TASK segment in the exact frozen ChatML contract format."""

    try:
        task_text = task["text"]
        visible_test = task["test_list"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("MBPP task must contain text and a non-empty test_list") from exc
    if not isinstance(task_text, str) or not isinstance(visible_test, str):
        raise ValueError("MBPP task text and first test must be strings")

    from rolekv.roles import task_segment

    utils_py = build_utils_py(task, training_rows, ctx_funcs=ctx_funcs)
    return task_segment(utils_py, task_text, visible_test)


def load_mbpp_splits(*, eval_count: int = EVAL_TASK_COUNT, eval_seed: int = EVAL_SEED) -> MBPPSplits:
    """Load MBPP's full configuration with the project's frozen split policy.

    The import is local so modules that only need prompt helpers can be imported
    without installing the optional Hugging Face dependency.
    """

    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "MBPP loading requires the 'datasets' package. Install the project dependencies first."
        ) from exc

    dataset = load_dataset(MBPP_DATASET, MBPP_CONFIG)
    missing = [name for name in (*TRAIN_SPLITS, TEST_SPLIT) if name not in dataset]
    if missing:
        raise ValueError(f"MBPP {MBPP_CONFIG!r} is missing required splits: {', '.join(missing)}")

    training_parts = [dataset[name] for name in TRAIN_SPLITS]
    for name, part in zip(TRAIN_SPLITS, training_parts, strict=True):
        if not len(part):
            raise ValueError(f"MBPP {name!r} split is empty")

    training = concatenate_datasets(training_parts)
    evaluation = select_evaluation_tasks(dataset[TEST_SPLIT], count=eval_count, seed=eval_seed)

    training_ids = _task_ids(training)
    evaluation_ids = _task_ids(evaluation)
    leaked_ids = training_ids & evaluation_ids
    if leaked_ids:
        preview = ", ".join(map(str, sorted(leaked_ids, key=str)[:5]))
        raise ValueError(f"MBPP train/eval leakage detected for task_id(s): {preview}")

    return MBPPSplits(training=training, evaluation=evaluation)
