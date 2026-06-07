#!/usr/bin/env python3
"""Build a qpos LeRobot dataset from an existing ee_pose dataset (no re-teleop).

Copies meta + data parquets, replaces ``actions`` with IK(post-step ee_pose) using
``observation.state[:6]`` as the arm seed (same convention as MyEnv absolute ee_pose).

IK is approximate for redundant arms: recovered qpos may not match an original qpos
recording bit-for-bit. Prefer recording qpos directly or using qpos_to_ee_pose_dataset.py
for the ee_pose copy. Does not modify the source tree.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_SRC))

from core.action_convert import EePoseToQposIK  # noqa: E402
from core.dataset_config import (  # noqa: E402
    XML_PATH,
    teleop_ee_pose_root,
    teleop_qpos_root,
)
from lerobot.datasets.compute_stats import get_feature_stats  # noqa: E402
from lerobot.datasets.utils import load_stats, write_stats  # noqa: E402


def _list_data_parquets(root: Path) -> list[Path]:
    data_dir = root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing {data_dir}")
    paths = sorted(data_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet under {data_dir}")
    return paths


def _copy_meta(src: Path, dst: Path) -> None:
    src_meta = src / "meta"
    dst_meta = dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    for name in (
        "info.json",
        "tasks.parquet",
        "tasks.jsonl",
        "episodes.jsonl",
        "stats.json",
    ):
        src_file = src_meta / name
        if src_file.is_file():
            shutil.copy2(src_file, dst_meta / name)

    ep_dir = src_meta / "episodes"
    if ep_dir.is_dir():
        shutil.copytree(ep_dir, dst_meta / "episodes", dirs_exist_ok=True)

    ep_stats = src_meta / "episodes_stats.jsonl"
    if ep_stats.is_file():
        shutil.copy2(ep_stats, dst_meta / "episodes_stats.jsonl")


def _link_or_copy_videos(src: Path, dst: Path, symlink: bool) -> None:
    src_videos = src / "videos"
    if not src_videos.is_dir():
        return
    dst_videos = dst / "videos"
    if dst_videos.exists() or dst_videos.is_symlink():
        if dst_videos.is_symlink():
            dst_videos.unlink()
        else:
            shutil.rmtree(dst_videos)
    if symlink:
        dst_videos.symlink_to(src_videos.resolve(), target_is_directory=True)
    else:
        shutil.copytree(src_videos, dst_videos)


def _convert_parquet(
    src_path: Path, dst_path: Path, ik: EePoseToQposIK
) -> tuple[int, float]:
    df = pd.read_parquet(src_path)
    ee_actions = np.stack([np.asarray(v, dtype=np.float32) for v in df["actions"]])
    pre_states = np.stack(
        [np.asarray(v, dtype=np.float32) for v in df["observation.state"]]
    )
    qpos_actions, ik_errs = ik.episode(ee_actions, pre_states)
    df = df.copy()
    df["actions"] = [row for row in qpos_actions]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)
    return len(df), float(ik_errs.max())


def _recompute_action_stats(dst: Path) -> None:
    chunks: list[np.ndarray] = []
    for pq in _list_data_parquets(dst):
        df = pd.read_parquet(pq, columns=["actions"])
        chunks.append(
            np.stack([np.asarray(v, dtype=np.float32) for v in df["actions"]])
        )
    actions = np.concatenate(chunks, axis=0)
    action_stats = get_feature_stats(actions, axis=0, keepdims=False)

    stats = load_stats(dst)
    if stats is None:
        stats = {}
    stats["actions"] = action_stats
    write_stats(stats, dst)


def _patch_info_note(dst: Path) -> None:
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["action_label"] = "qpos"
    info["derived_from_ee_pose"] = True
    info_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Source ee_pose LeRobot root (default: teleop_ee_pose_root)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Output qpos LeRobot root (default: teleop_qpos_root)",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=XML_PATH,
        help="MuJoCo scene XML for IK (default: dataset_config.XML_PATH)",
    )
    parser.add_argument(
        "--symlink-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symlink videos/ from src (default: true)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = Path(os.path.expanduser(args.src) if args.src else teleop_ee_pose_root())
    dst_root = Path(os.path.expanduser(args.dst) if args.dst else teleop_qpos_root())

    if not (src_root / "meta" / "info.json").is_file():
        raise SystemExit(f"Not a LeRobot dataset: {src_root}")

    if dst_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst_root} exists; pass --overwrite to replace")
        if dst_root.resolve() == src_root.resolve():
            raise SystemExit("dst must differ from src")
        if dst_root.is_symlink():
            dst_root.unlink()
        else:
            shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)
    _copy_meta(src_root, dst_root)
    _link_or_copy_videos(src_root, dst_root, symlink=args.symlink_videos)

    ik = EePoseToQposIK(args.xml)
    total_rows = 0
    max_ik_err = 0.0
    for src_pq in _list_data_parquets(src_root):
        rel = src_pq.relative_to(src_root / "data")
        dst_pq = dst_root / "data" / rel
        n, ep_max_err = _convert_parquet(src_pq, dst_pq, ik)
        total_rows += n
        max_ik_err = max(max_ik_err, ep_max_err)
        print(f"  {rel}: {n} rows, max ik_err={ep_max_err:.5f}")

    _recompute_action_stats(dst_root)
    _patch_info_note(dst_root)

    print(f"Source: {src_root}")
    print(f"Output: {dst_root}")
    print(f"Converted {total_rows} frames (actions: ee_pose -> qpos IK)")
    print(f"Max IK residual across files: {max_ik_err:.5f}")
    if args.symlink_videos:
        print(f"Videos: symlink -> {src_root / 'videos'}")


if __name__ == "__main__":
    main()
