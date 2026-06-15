#!/usr/bin/env python3
"""Quantify j1 delta bias in first K steps after each replan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _episode_rows(ep_dir: Path) -> list[dict]:
    rows_path = ep_dir / "action_chunks.jsonl"
    if not rows_path.exists():
        return []
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _collect(trace_dir: Path, horizon: int) -> tuple[dict[int, list[float]], int]:
    vals = {k: [] for k in range(1, horizon + 1)}
    num_replans = 0
    for ep in sorted(trace_dir.glob("episode_*")):
        rows = _episode_rows(ep)
        for row in rows:
            num_replans += 1
            delta_chunk = np.asarray(row["arm_delta_chunk"], dtype=np.float64)[:, 1]
            for k in range(1, horizon + 1):
                if delta_chunk.shape[0] >= k:
                    vals[k].append(float(delta_chunk[k - 1]))
    return vals, num_replans


def _summarize(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "neg_frac": float((arr < 0.0).mean()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace_dirs",
        nargs="+",
        help="Trace directories that contain episode_*/action_chunks.jsonl",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="How many post-replan steps to analyze (default: 3)",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Emit JSON instead of pretty text",
    )
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError(f"--horizon must be >= 1, got {args.horizon}")

    report: dict[str, dict] = {}
    for trace in args.trace_dirs:
        trace_dir = Path(trace)
        vals, num_replans = _collect(trace_dir, args.horizon)
        per_step: dict[str, dict] = {}
        pooled = []
        for k in range(1, args.horizon + 1):
            arr = np.asarray(vals[k], dtype=np.float64)
            if arr.size == 0:
                continue
            per_step[f"step_{k}"] = _summarize(arr)
            pooled.append(arr)
        if pooled:
            pooled_arr = np.concatenate(pooled, axis=0)
            report[str(trace_dir)] = {
                "num_replans": int(num_replans),
                "per_step": per_step,
                "pooled_first_horizon": _summarize(pooled_arr),
            }

    if args.as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    for trace_name, data in report.items():
        print(f"=== {trace_name} ===")
        print(f"num_replans: {data['num_replans']}")
        for step_name, stats in data["per_step"].items():
            print(
                f"{step_name}: mean={stats['mean']:+.6f} median={stats['median']:+.6f} "
                f"neg_frac={stats['neg_frac']:.3f} p10={stats['p10']:+.6f} "
                f"p90={stats['p90']:+.6f} n={stats['n']}"
            )
        pooled = data["pooled_first_horizon"]
        print(
            f"pooled_first_horizon: mean={pooled['mean']:+.6f} "
            f"neg_frac={pooled['neg_frac']:.3f} n={pooled['n']}"
        )
        print()


if __name__ == "__main__":
    main()
