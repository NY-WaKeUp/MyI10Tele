#!/usr/bin/env python3
"""Compare per-step EE motion in LeRobot ee_pose data vs openpi eval trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def wrap_pi(angles: np.ndarray) -> np.ndarray:
    return ((np.asarray(angles, dtype=np.float64) + np.pi) % (2 * np.pi)) - np.pi


def pose_delta_6d(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = np.asarray(target, dtype=np.float64)[:6] - np.asarray(ref, dtype=np.float64)[:6]
    d[3:6] = wrap_pi(d[3:6])
    return d


def summarize_norms(name: str, xyz: np.ndarray, pose6: np.ndarray) -> None:
    print(f"\n=== {name} (n={len(xyz)}) ===")
    for label, arr in [("xyz norm (m)", xyz), ("6D wrapped norm", pose6)]:
        print(
            f"  {label}: mean={arr.mean():.5f}  median={np.median(arr):.5f}  "
            f"p90={np.quantile(arr, 0.9):.5f}  max={arr.max():.5f}"
        )
    print(f"  xyz < 5mm frac: {(xyz < 0.005).mean():.2%}")
    print(f"  xyz < 2cm frac: {(xyz < 0.02).mean():.2%}")


def load_dataset_ee_deltas(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Dataset: state=qpos pre, actions=absolute ee post (same as 0.tele.py ee_pose)."""
    data_dir = data_root / "data"
    xyz_list: list[np.ndarray] = []
    pose6_list: list[np.ndarray] = []

    for ep_path in sorted(data_dir.rglob("episode_*.parquet")):
        df = pd.read_parquet(ep_path, columns=["observation.state", "actions"])
        states = np.stack(df["observation.state"].to_numpy())  # qpos pre (not used below)
        actions = np.stack(df["actions"].to_numpy())  # absolute ee post

        # Proxy for "how much the demo moves per frame": post_{t} vs post_{t-1}.
        # If the arm tracks well, ee_pre_t ≈ actions_{t-1}.
        d_xyz = []
        d6 = []
        for t in range(1, len(actions)):
            delta = pose_delta_6d(actions[t], actions[t - 1])
            d6.append(np.linalg.norm(delta))
            d_xyz.append(np.linalg.norm(delta[:3]))
        xyz_list.append(np.asarray(d_xyz))
        pose6_list.append(np.asarray(d6))

        # Command-style delta: ||ee_post_t - ee_post_{t-1}|| is the demo step size.
        del states

    return np.concatenate(xyz_list), np.concatenate(pose6_list)


def load_trace_xyz(trace_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """From rollout.npz written by 8.val_openpi_sim.py (wrapped ee metrics)."""
    xyz_parts = []
    motion_parts = []
    for npz_path in sorted(trace_dir.glob("episode_*/rollout.npz")):
        z = np.load(npz_path)
        if "ee_xyz_cmd_norm" not in z or len(z["ee_xyz_cmd_norm"]) == 0:
            continue
        xyz_parts.append(z["ee_xyz_cmd_norm"])
        if "ee_motion_norm" in z:
            motion_parts.append(z["ee_motion_norm"])
    if not xyz_parts:
        raise SystemExit(f"No ee_xyz_cmd_norm in {trace_dir} (re-run eval with updated trace code)")
    xyz = np.concatenate(xyz_parts)
    motion = np.concatenate(motion_parts) if motion_parts else np.zeros_like(xyz)
    return xyz, motion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("~/MyI10Tele/data_auboI10_ee_pose_v20").expanduser(),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("~/MyI10Tele/openpi_eval_trace_v2").expanduser(),
    )
    args = parser.parse_args()

    print(f"Dataset: {args.data_root}")
    print(f"Trace:   {args.trace_dir}")

    ds_xyz, ds_pose6 = load_dataset_ee_deltas(args.data_root)
    summarize_norms("Dataset (|ee_post_t - ee_post_{t-1}|, wrapped)", ds_xyz, ds_pose6)

    tr_xyz, tr_motion = load_trace_xyz(args.trace_dir)
    summarize_norms("Trace policy cmd (|ee_cmd - ee_pre|, xyz)", tr_xyz, tr_xyz)
    summarize_norms("Trace sim motion (|ee_post - ee_pre|, 6D wrapped)", tr_motion, tr_motion)

    ratio = tr_xyz.mean() / ds_xyz.mean() if ds_xyz.mean() > 0 else float("nan")
    print(f"\n>>> Mean xyz cmd / mean dataset step: {ratio:.3f}x")
    print(
        ">>> If trace ee_motion << trace xyz_cmd, IK/sim is not reaching commanded targets."
    )

    summaries = sorted(args.trace_dir.glob("episode_*/summary.json"))
    if summaries:
        print("\n=== Trace episode summaries ===")
        for p in summaries:
            s = json.loads(p.read_text(encoding="utf-8"))
            print(
                f"  ep {s['episode_index']:03d}: xyz_cmd={s.get('ee_xyz_cmd_norm_mean', 0):.4f} "
                f"ee_motion={s.get('ee_motion_norm_mean', 0):.5f} "
                f"grip_toggles={s.get('gripper_toggles', 0)} ok={s['success']}"
            )


if __name__ == "__main__":
    main()
