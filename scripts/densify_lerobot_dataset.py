#!/usr/bin/env python3
"""Densify a LeRobot qpos dataset by interpolating skipped teleop segments.

Original v20 data may drop frames when the arm barely moves; consecutive rows still
use 0.05 s timestamps but joint deltas can be large. This script rebuilds episodes at
the same fps with linear arm interpolation, step-wise gripper for 0/1 toggles, and
blended camera frames between source keyframes.

Does not modify the source tree. Output is a new dataset root suitable for training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Project imports when run from repo root.
_REPO_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_SRC))

from core.dataset_config import REPO_NAME, TASK_NAME, dataset_root  # noqa: E402


def _chw_float_to_hwc_uint8(img: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        scale = 255.0 if float(arr.max()) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
    return arr


def _blend_hwc_uint8(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    fa = a.astype(np.float32)
    fb = b.astype(np.float32)
    out = (1.0 - alpha) * fa + alpha * fb
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _interp_qpos(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    arm = (1.0 - alpha) * q0[:6] + alpha * q1[:6]
    if abs(float(q1[6] - q0[6])) > 0.5:
        grip = q0[6] if alpha < 0.5 else q1[6]
    else:
        grip = (1.0 - alpha) * q0[6] + alpha * q1[6]
    return np.array([*arm, grip], dtype=np.float32)


def densify_episode_actions(
    actions: np.ndarray, max_arm_step: float
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Return dense post-step qpos keypoints and (left_image_index, alpha) tags."""
    n = len(actions)
    if n == 0:
        return actions, []
    if n == 1:
        return actions.copy(), [(0, 0.0)]

    dense_posts: list[np.ndarray] = [actions[0].astype(np.float32, copy=True)]
    image_tags: list[tuple[int, float]] = [(0, 0.0)]

    for i in range(n - 1):
        q0, q1 = actions[i], actions[i + 1]
        arm_delta = float(np.linalg.norm(q1[:6] - q0[:6]))
        n_seg = max(1, int(np.ceil(arm_delta / max_arm_step)))
        for j in range(1, n_seg + 1):
            alpha = j / n_seg
            dense_posts.append(_interp_qpos(q0, q1, alpha))
            image_tags.append((i, alpha))

    return np.stack(dense_posts, axis=0), image_tags


def _load_episode_rows(src: LeRobotDataset, episode_index: int):
    ep = src.hf_dataset.filter(lambda x, e=episode_index: x["episode_index"] == e)
    n = len(ep)
    indices = [int(ep[i]["index"]) for i in range(n)]
    actions = np.array(ep["actions"], dtype=np.float32)
    obj_init = np.asarray(ep[0]["obj_init"], dtype=np.float32)
    return indices, actions, obj_init


def _report_episode(
    name: str, before: int, after: int, actions: np.ndarray, dense: np.ndarray
) -> None:
    if before < 2 or len(dense) < 2:
        return
    d0 = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)
    d1 = np.linalg.norm(np.diff(dense[:, :6], axis=0), axis=1)
    print(
        f"  {name}: frames {before} -> {after} | "
        f"max |dq_arm| {d0.max():.4f} -> {d1.max():.4f} rad"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Source LeRobot root (default: dataset_config qpos path)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=os.path.expanduser("~/MyI10Tele/data_auboI10_qpos_v20_interp"),
        help="Output LeRobot root (must not exist unless --overwrite)",
    )
    parser.add_argument(
        "--max-arm-step",
        type=float,
        default=0.015,
        help="Max L2 norm (rad) between consecutive arm qpos rows at 20 Hz",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Process only the first N episodes (for smoke tests)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = os.path.expanduser(args.src) if args.src else dataset_root()
    dst_root = os.path.expanduser(args.dst)

    if os.path.exists(dst_root):
        if not args.overwrite:
            raise SystemExit(f"{dst_root} exists; pass --overwrite to replace")
        import shutil

        shutil.rmtree(dst_root)

    src = LeRobotDataset(REPO_NAME, root=src_root)
    info = json.loads(
        open(os.path.join(src_root, "meta", "info.json"), encoding="utf-8").read()
    )
    fps = int(info["fps"])
    raw_features = info["features"]
    features = {}
    for key, feat in raw_features.items():
        if key in (
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ):
            continue
        feat = dict(feat)
        if "shape" in feat:
            feat["shape"] = tuple(feat["shape"])
        features[key] = feat
    robot_type = info.get("robot_type", "aubo_i10_inspire")
    vcodec = info["features"]["observation.image"]["info"].get("video.codec", "h264")
    if sys.platform == "darwin" and vcodec == "h264":
        vcodec = "auto"

    n_ep = src.num_episodes
    if args.max_episodes is not None:
        n_ep = min(n_ep, args.max_episodes)

    print(f"Source: {src_root}")
    print(f"Output: {dst_root}")
    print(f"Episodes: {n_ep}, fps={fps}, max_arm_step={args.max_arm_step} rad")

    dst = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=dst_root,
        robot_type=robot_type,
        fps=fps,
        vcodec=vcodec,
        streaming_encoding=True,
        encoder_queue_maxsize=90,
        features=features,
        image_writer_threads=6,
        image_writer_processes=0 if sys.platform == "darwin" else 5,
    )

    total_in = 0
    total_out = 0

    for ep_idx in range(n_ep):
        indices, actions, obj_init = _load_episode_rows(src, ep_idx)
        n_in = len(indices)
        total_in += n_in

        images_agent = [
            _chw_float_to_hwc_uint8(src[i]["observation.image"]) for i in indices
        ]
        images_wrist = [
            _chw_float_to_hwc_uint8(src[i]["observation.wrist_image"]) for i in indices
        ]

        dense_posts, image_tags = densify_episode_actions(actions, args.max_arm_step)
        n_dense = len(dense_posts)
        n_out = n_dense - 1
        total_out += n_out
        _report_episode(f"ep{ep_idx}", n_in, n_out, actions, dense_posts)

        for k in range(n_out):
            li, a0 = image_tags[k]
            _, a1 = image_tags[k + 1]
            alpha_img = 0.5 * (a0 + a1)
            ri = li + 1
            agent = _blend_hwc_uint8(images_agent[li], images_agent[ri], alpha_img)
            wrist = _blend_hwc_uint8(images_wrist[li], images_wrist[ri], alpha_img)
            dst.add_frame(
                {
                    "observation.image": agent,
                    "observation.wrist_image": wrist,
                    "observation.state": dense_posts[k],
                    "actions": dense_posts[k + 1],
                    "obj_init": obj_init,
                    "task": TASK_NAME,
                }
            )

        dst.save_episode(parallel_encoding=sys.platform != "darwin")

    dst.stop_image_writer()
    dst.finalize()

    print(
        f"Done. frames {total_in} -> {total_out} ({total_out / max(total_in, 1):.2f}x)"
    )
    print(f"Point dataset_config.AUBOI10_QPOS_ROOT to:\n  {dst_root}")


if __name__ == "__main__":
    main()
