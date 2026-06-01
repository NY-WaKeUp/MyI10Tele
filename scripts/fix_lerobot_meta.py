#!/usr/bin/env python3
"""Write meta/tasks.jsonl and meta/episodes.jsonl for openpi / lerobot 0.1.x loaders.

LeRobot v2.0 datasets created with newer lerobot may only have tasks.parquet and
episodes under meta/episodes/*.parquet. openpi's bundled lerobot expects jsonl files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _write_tasks_jsonl(meta_dir: Path, task_override: str | None = None) -> int:
    tasks_parquet = meta_dir / "tasks.parquet"
    if tasks_parquet.exists():
        tasks_df = pd.read_parquet(tasks_parquet)
        with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
            for task, row in tasks_df.iterrows():
                f.write(
                    json.dumps(
                        {"task_index": int(row["task_index"]), "task": str(task)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(tasks_df)
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    task_name = task_override or str(
        info.get("task") or info.get("default_task") or "Put cube on the black platform"
    )
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": task_name}) + "\n")
    return 1


def _default_task_name(meta_dir: Path) -> str:
    tasks_jsonl = meta_dir / "tasks.jsonl"
    if tasks_jsonl.exists():
        first = json.loads(tasks_jsonl.read_text(encoding="utf-8").splitlines()[0])
        return str(first["task"])
    return "default_task"


def _write_episodes_jsonl_from_data_parquets(dataset_root: Path, meta_dir: Path) -> int:
    """LeRobot v2.0: only per-episode data parquet, no meta/episodes/*.parquet."""
    task_name = _default_task_name(meta_dir)
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"No data/ under {dataset_root}")
    episode_paths = sorted(data_root.rglob("episode_*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.parquet under {data_root}")

    records: list[dict] = []
    for ep_path in episode_paths:
        stem = ep_path.stem  # episode_000042
        episode_index = int(stem.split("_", 1)[1])
        df = pd.read_parquet(ep_path, columns=["episode_index"])
        length = int(len(df))
        records.append(
            {
                "episode_index": episode_index,
                "tasks": [task_name],
                "length": length,
            }
        )
    records.sort(key=lambda r: r["episode_index"])
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def _write_episodes_jsonl(dataset_root: Path, meta_dir: Path) -> int:
    ep_dir = meta_dir / "episodes"
    if ep_dir.is_dir():
        dfs = [pd.read_parquet(p) for p in sorted(ep_dir.rglob("*.parquet"))]
        ep = pd.concat(dfs, ignore_index=True)
    elif (meta_dir / "episodes.parquet").exists():
        ep = pd.read_parquet(meta_dir / "episodes.parquet")
    else:
        return _write_episodes_jsonl_from_data_parquets(dataset_root, meta_dir)

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for _, row in ep.sort_values("episode_index").iterrows():
            tasks = row["tasks"]
            if hasattr(tasks, "tolist"):
                tasks = tasks.tolist()
            rec = {
                "episode_index": int(row["episode_index"]),
                "tasks": tasks,
                "length": int(row["length"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(ep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="LeRobot dataset root (contains meta/info.json)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Put cube on the black platform",
        help="Task string when meta/tasks.parquet is missing (default: Put cube on the black platform)",
    )
    args = parser.parse_args()
    meta_dir = args.dataset_root / "meta"
    if not (meta_dir / "info.json").exists():
        raise SystemExit(f"Not a LeRobot dataset: {args.dataset_root}")

    n_tasks = _write_tasks_jsonl(meta_dir, task_override=args.task)
    n_ep = _write_episodes_jsonl(args.dataset_root, meta_dir)
    print(f"Wrote {meta_dir / 'tasks.jsonl'} ({n_tasks} tasks)")
    print(f"Wrote {meta_dir / 'episodes.jsonl'} ({n_ep} episodes)")


if __name__ == "__main__":
    main()
