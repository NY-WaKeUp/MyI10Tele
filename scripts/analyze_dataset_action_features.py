#!/usr/bin/env python3
"""Analyze LeRobot qpos dataset action features and sag-bias hypothesis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


JOINT_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"]


def _to_vec7(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 7:
        raise ValueError(f"Expected 7D vector, got shape {arr.shape}")
    return arr


def _format_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    df = pd.DataFrame(matrix, index=JOINT_NAMES, columns=JOINT_NAMES)
    df.to_csv(path, float_format="%.10f")


def _load_dataset_arrays(
    root: Path,
    episode_start: int | None,
    episode_end: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parquets = sorted((root / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet found under {(root / 'data')}")

    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    if "episode_index" not in df.columns:
        raise ValueError("Missing episode_index column in dataset parquet")
    if "observation.state" not in df.columns or "actions" not in df.columns:
        raise ValueError("Missing observation.state/actions columns in dataset parquet")

    if episode_start is not None:
        df = df.loc[df["episode_index"] >= episode_start]
    if episode_end is not None:
        df = df.loc[df["episode_index"] <= episode_end]
    if df.empty:
        raise ValueError("No frames after episode range filtering")

    if "frame_index" in df.columns:
        df = df.sort_values(["episode_index", "frame_index"])
    else:
        df = df.sort_values(["episode_index"])

    states = np.stack([_to_vec7(v) for v in df["observation.state"].values], axis=0)
    actions = np.stack([_to_vec7(v) for v in df["actions"].values], axis=0)
    episode_ids = df["episode_index"].to_numpy(dtype=np.int64)
    if "frame_index" in df.columns:
        frame_ids = df["frame_index"].to_numpy(dtype=np.int64)
    else:
        frame_ids = np.arange(len(df), dtype=np.int64)
    return states, actions, episode_ids, frame_ids


def _temporal_delta_from_state(
    states: np.ndarray, episode_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros_like(states, dtype=np.float64)
    valid = np.zeros(states.shape[0], dtype=bool)
    unique_eps = np.unique(episode_ids)
    for ep in unique_eps:
        idx = np.where(episode_ids == ep)[0]
        if idx.size < 2:
            continue
        delta[idx[:-1]] = states[idx[1:]] - states[idx[:-1]]
        valid[idx[:-1]] = True
    return delta, valid


def _compute_metrics(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    move_thresh: float,
    near_zero_thresh: float,
    smooth_majority_threshold: float,
    sag_dom_threshold: float,
) -> dict:
    delta_label = actions - states
    delta_temporal, temporal_valid = _temporal_delta_from_state(states, episode_ids)

    label_arm_l2 = np.linalg.norm(delta_label[:, :6], axis=1)
    label_signal = float(np.percentile(label_arm_l2, 95))
    use_temporal = label_signal < move_thresh * 0.1

    if use_temporal:
        delta = delta_temporal[temporal_valid]
    else:
        delta = delta_label

    arm_delta = delta[:, :6]
    arm_abs = np.abs(arm_delta)
    arm_l2 = np.linalg.norm(arm_delta, axis=1)
    moving_mask = arm_l2 > move_thresh
    near_zero_mask = arm_l2 < near_zero_thresh

    dominant_dim = np.argmax(arm_abs, axis=1)
    dom_counts = np.bincount(dominant_dim, minlength=6).astype(np.float64)
    dom_frac_all = dom_counts / float(len(dominant_dim))

    moving_dom = dominant_dim[moving_mask]
    moving_dom_counts = np.bincount(moving_dom, minlength=6).astype(np.float64)
    moving_dom_frac = moving_dom_counts / max(float(moving_dom.size), 1.0)

    j1_neg = arm_delta[:, 1] < 0.0
    j1_neg_dom_mask = moving_mask & j1_neg & (dominant_dim == 1)
    j1_neg_frac_moving = float(j1_neg[moving_mask].mean()) if moving_mask.any() else 0.0
    j1_neg_dom_frac_moving = (
        float(j1_neg_dom_mask.sum()) / float(moving_mask.sum())
        if moving_mask.any()
        else 0.0
    )

    per_dim_stats = []
    for i in range(6):
        d = arm_delta[:, i]
        ad = np.abs(d)
        per_dim_stats.append(
            {
                "joint": JOINT_NAMES[i],
                "mean": float(d.mean()),
                "std": float(d.std(ddof=0)),
                "mean_abs": float(ad.mean()),
                "p10": float(np.percentile(d, 10)),
                "p50": float(np.percentile(d, 50)),
                "p90": float(np.percentile(d, 90)),
                "neg_frac": float((d < 0.0).mean()),
                "dominant_frac_all": float(dom_frac_all[i]),
                "dominant_frac_moving": float(moving_dom_frac[i]),
            }
        )

    max_other_dom = (
        float(np.max(np.delete(moving_dom_frac, 1))) if moving_dom_frac.size else 0.0
    )
    smooth_majority = float(near_zero_mask.mean()) >= smooth_majority_threshold
    sag_dominant = j1_neg_dom_frac_moving >= sag_dom_threshold and (
        moving_dom_frac[1] > max_other_dom
    )
    if smooth_majority and sag_dominant:
        verdict = "supported"
    elif smooth_majority or sag_dominant:
        verdict = "partially_supported"
    else:
        verdict = "not_supported"

    return {
        "num_frames": int(states.shape[0]),
        "num_episodes": int(np.unique(states.shape[0] * [0]).size),  # overwritten later
        "delta_source": (
            "temporal_state_diff" if use_temporal else "label_action_minus_state"
        ),
        "label_delta_arm_l2_p95": label_signal,
        "move_threshold": float(move_thresh),
        "near_zero_threshold": float(near_zero_thresh),
        "arm_move_frac": float(moving_mask.mean()),
        "arm_near_zero_frac": float(near_zero_mask.mean()),
        "j1_negative_frac_moving": j1_neg_frac_moving,
        "j1_negative_dominant_frac_moving": j1_neg_dom_frac_moving,
        "dominant_frac_moving": {
            JOINT_NAMES[i]: float(moving_dom_frac[i]) for i in range(6)
        },
        "per_joint_delta_stats": per_dim_stats,
        "hypothesis_checks": {
            "smooth_majority": bool(smooth_majority),
            "sag_dominant": bool(sag_dominant),
            "smooth_majority_threshold": float(smooth_majority_threshold),
            "sag_dominant_threshold": float(sag_dom_threshold),
            "max_other_dominant_frac_moving": max_other_dom,
        },
        "hypothesis_verdict": verdict,
        "cov_action": np.cov(actions, rowvar=False).tolist(),
        "cov_delta_label": np.cov(delta_label, rowvar=False).tolist(),
        "cov_delta_used": np.cov(delta, rowvar=False).tolist(),
    }


def _render_md(
    dataset_root: Path,
    episode_start: int | None,
    episode_end: int | None,
    metrics: dict,
    json_path: Path,
    cov_action_csv: Path,
    cov_delta_csv: Path,
    cov_delta_label_csv: Path,
) -> str:
    header = [
        "# 数据集动作特征分析",
        "",
        "## 配置",
        f"- dataset_root: `{dataset_root}`",
        f"- episode_range: `{episode_start}` ~ `{episode_end}`",
        f"- num_frames: `{metrics['num_frames']}`",
        f"- move_threshold (arm L2): `{metrics['move_threshold']}`",
        f"- near_zero_threshold (arm L2): `{metrics['near_zero_threshold']}`",
        f"- delta_source_used: `{metrics['delta_source']}`",
        f"- label_delta_arm_l2_p95: `{metrics['label_delta_arm_l2_p95']:.6e}`",
        "",
        "## 你的假设检验（平稳多 + 下坠模式被学到）",
        "",
        f"- hypothesis_verdict: **`{metrics['hypothesis_verdict']}`**",
        f"- arm_near_zero_frac: `{metrics['arm_near_zero_frac']:.4f}`",
        f"- j1_negative_frac_moving: `{metrics['j1_negative_frac_moving']:.4f}`",
        f"- j1_negative_dominant_frac_moving: `{metrics['j1_negative_dominant_frac_moving']:.4f}`",
        "",
        "### moving 区间主导维度占比（argmax |delta|）",
        "",
        "| joint | dominant_frac_moving |",
        "|---|---:|",
    ]
    for j, v in metrics["dominant_frac_moving"].items():
        header.append(f"| {j} | {v:.4f} |")

    header.extend(
        [
            "",
            "### 各关节 delta 统计（arm 6维）",
            "",
            "| joint | mean | std | mean_abs | neg_frac | p10 | p50 | p90 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["per_joint_delta_stats"]:
        header.append(
            f"| {row['joint']} | {row['mean']:+.6f} | {row['std']:.6f} | "
            f"{row['mean_abs']:.6f} | {row['neg_frac']:.4f} | "
            f"{row['p10']:+.6f} | {row['p50']:+.6f} | {row['p90']:+.6f} |"
        )

    header.extend(
        [
            "",
            "## 协方差矩阵输出",
            "",
            f"- absolute action covariance csv: `{cov_action_csv}`",
            f"- delta(used for analysis) covariance csv: `{cov_delta_csv}`",
            f"- delta(action-state label) covariance csv: `{cov_delta_label_csv}`",
            f"- full metrics json: `{json_path}`",
        ]
    )
    return "\n".join(header) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-root",
        type=str,
        default="~/MyI10Tele/data_auboI10_qpos_v21_continuous_correctobjinit",
        help="LeRobot dataset root",
    )
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument(
        "--move-threshold",
        type=float,
        default=0.002,
        help="Arm delta L2 threshold to mark moving steps",
    )
    parser.add_argument(
        "--near-zero-threshold",
        type=float,
        default=0.0005,
        help="Arm delta L2 threshold to mark near-static steps",
    )
    parser.add_argument(
        "--smooth-majority-threshold",
        type=float,
        default=0.6,
        help="Threshold for near-static ratio in hypothesis check",
    )
    parser.add_argument(
        "--sag-dominant-threshold",
        type=float,
        default=0.25,
        help="Threshold for j1 negative dominant ratio in moving steps",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="dataset_action_features",
        help="Output file prefix under --output-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory",
    )
    args = parser.parse_args()

    dataset_root = Path(os.path.expanduser(args.lerobot_root))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    states, actions, episode_ids, _ = _load_dataset_arrays(
        dataset_root, args.episode_start, args.episode_end
    )
    metrics = _compute_metrics(
        states=states,
        actions=actions,
        episode_ids=episode_ids,
        move_thresh=args.move_threshold,
        near_zero_thresh=args.near_zero_threshold,
        smooth_majority_threshold=args.smooth_majority_threshold,
        sag_dom_threshold=args.sag_dominant_threshold,
    )
    metrics["num_episodes"] = int(np.unique(episode_ids).size)

    cov_action = np.asarray(metrics["cov_action"], dtype=np.float64)
    cov_delta = np.asarray(metrics["cov_delta_used"], dtype=np.float64)

    cov_action_csv = output_dir / f"{args.output_prefix}_cov_action.csv"
    cov_delta_csv = output_dir / f"{args.output_prefix}_cov_delta_used.csv"
    cov_delta_label_csv = output_dir / f"{args.output_prefix}_cov_delta_label.csv"
    _format_matrix_csv(cov_action_csv, cov_action)
    _format_matrix_csv(cov_delta_csv, cov_delta)
    _format_matrix_csv(
        cov_delta_label_csv, np.asarray(metrics["cov_delta_label"], dtype=np.float64)
    )

    json_path = output_dir / f"{args.output_prefix}.json"
    json_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_text = _render_md(
        dataset_root=dataset_root,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        metrics=metrics,
        json_path=json_path,
        cov_action_csv=cov_action_csv,
        cov_delta_csv=cov_delta_csv,
        cov_delta_label_csv=cov_delta_label_csv,
    )
    md_path = output_dir / f"{args.output_prefix}.md"
    md_path.write_text(md_text, encoding="utf-8")

    print(f"Saved report: {md_path}")
    print(f"Saved metrics: {json_path}")
    print(
        f"Saved covariance CSV: {cov_action_csv}, {cov_delta_csv}, {cov_delta_label_csv}"
    )


if __name__ == "__main__":
    main()
