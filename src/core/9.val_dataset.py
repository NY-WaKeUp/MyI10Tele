#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline dataset validation: policy predictions vs recorded actions + plots.

Compare model outputs on LeRobot dataset frames (no MuJoCo). Visualization style
follows ``1.replay_dataset.py``.

Examples
--------
ACT checkpoint on interpolated qpos dataset::

    cd ~/MyI10Tele
    PYTHONPATH=src python src/core/9.val_dataset.py \\
        --policy act \\
        --checkpoint .ckpt/auboI10_act_w_2_view_temporal_ensemble_coeff09 \\
        --episodes 0,1,2 \\
        --output-dir ./val_dataset_act

PI0 checkpoint::

    PYTHONPATH=src python src/core/9.val_dataset.py \\
        --policy pi0 \\
        --checkpoint .ckpt/pi0_base/data_w_shadow_h264_znear0001 \\
        --episodes all \\
        --output-dir ./val_dataset_pi0

Re-plot saved npz without re-inference::

    PYTHONPATH=src python src/core/9.val_dataset.py \\
        --plot-only ./val_dataset_act/episode_000.npz \\
        --output-dir ./val_dataset_act/replot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from matplotlib.widgets import Slider
from tqdm.auto import tqdm

from core.dataset_config import ACTION_LABEL, REPO_NAME, dataset_root
from core.my_policy import MyPIPolicy, MyPolicy, load_paligemma_tokenizer

ACTION_DIM_LABELS = (
    ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    if ACTION_LABEL == "ee_pose"
    else [f"j{i}" for i in range(6)] + ["gripper"]
)
ARM_DIMS = 6
ACTION_DIM = 7


def _wrap_pi(angles: np.ndarray) -> np.ndarray:
    return ((np.asarray(angles, dtype=np.float64) + np.pi) % (2 * np.pi)) - np.pi


def _action_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-dim error; rpy uses shortest angular difference when ACTION_LABEL is ee_pose."""
    err = np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)
    if ACTION_LABEL == "ee_pose":
        err[:, 3:6] = _wrap_pi(err[:, 3:6])
    return err.astype(np.float32)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    err = _action_error(pred, gt)
    abs_err = np.abs(err)
    l2 = np.linalg.norm(err[:, :ARM_DIMS], axis=1)
    per_dim_mae = abs_err.mean(axis=0)
    per_dim_rmse = np.sqrt((err**2).mean(axis=0))
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae_arm": float(abs_err[:, :ARM_DIMS].mean()),
        "rmse_arm": float(np.sqrt((err[:, :ARM_DIMS] ** 2).mean())),
        "mae_gripper": float(abs_err[:, ARM_DIMS].mean()),
        "l2_arm_mean": float(l2.mean()),
        "l2_arm_max": float(l2.max()),
        "per_dim_mae": per_dim_mae.tolist(),
        "per_dim_rmse": per_dim_rmse.tolist(),
    }


@dataclass
class EpisodeResult:
    episode_index: int
    states: np.ndarray
    gt_actions: np.ndarray
    pred_actions: np.ndarray
    frame_indices: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return _action_error(self.pred_actions, self.gt_actions)

    def summary(self) -> dict:
        out = _metrics(self.pred_actions, self.gt_actions)
        out["episode_index"] = self.episode_index
        out["num_frames"] = int(len(self.gt_actions))
        return out


def _episode_frame_indices(dataset: LeRobotDataset, episode_id: int) -> np.ndarray:
    ep_table = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_id)
    return np.array(ep_table["index"], dtype=np.int64)


def _batch_from_item(item: dict, device: torch.device) -> dict:
    batch: dict = {}
    for key, val in item.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.unsqueeze(0).to(device)
        elif key == "task":
            batch[key] = [val] if isinstance(val, str) else val
    if "timestamp" not in batch and "frame_index" in item:
        batch["timestamp"] = torch.tensor(
            [float(item["frame_index"]) / 20.0], dtype=torch.float32, device=device
        )
    return batch


def load_act_policy(checkpoint: str, device: torch.device) -> ACTPolicy:
    root = dataset_root(interp=True)
    meta = LeRobotDatasetMetadata(REPO_NAME, root=root)
    in_feat, out_feat = MyPolicy.input_output_features_from_metadata(meta)
    cfg = ACTConfig(
        input_features=in_feat,
        output_features=out_feat,
        chunk_size=100,
        n_action_steps=1,
        temporal_ensemble_coeff=0.9,
        dropout=0.1,
        device="cpu",
    )
    policy = ACTPolicy.from_pretrained(checkpoint, config=cfg, dataset_stats=meta.stats)
    policy.to(device)
    policy.eval()
    return policy


def load_pi0_policy(checkpoint: str, device: torch.device):
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from transformers import AutoTokenizer

    root = dataset_root(interp=True)
    meta = LeRobotDatasetMetadata(REPO_NAME, root=root)
    in_feat, out_feat = MyPolicy.input_output_features_from_metadata(meta)
    cfg = PI0Config(
        input_features=in_feat,
        output_features=out_feat,
        compile_model=False,
        dtype="bfloat16",
        gradient_checkpointing=True,
        train_expert_only=True,
        chunk_size=100,
        n_action_steps=1,
        device=str(device),
    )
    policy = PI0Policy.from_pretrained(
        checkpoint, config=cfg, dataset_stats=meta.stats, local_files_only=True
    )
    policy.to(device)
    policy.eval()
    tokenizer = load_paligemma_tokenizer(allow_hub_download=False)
    tokenizer.padding_side = "right"
    pi_lang = MyPIPolicy(tokenizer, cfg.tokenizer_max_length)
    return policy, pi_lang


def infer_episode(
    policy,
    dataset: LeRobotDataset,
    episode_id: int,
    device: torch.device,
    *,
    pi_lang: MyPIPolicy | None = None,
    infer_mode: str = "select_action",
) -> EpisodeResult:
    indices = _episode_frame_indices(dataset, episode_id)
    states = np.zeros((len(indices), ACTION_DIM), dtype=np.float32)
    gt_actions = np.zeros((len(indices), ACTION_DIM), dtype=np.float32)
    pred_actions = np.zeros((len(indices), ACTION_DIM), dtype=np.float32)

    policy.reset()
    for i, abs_idx in enumerate(tqdm(indices, desc=f"ep {episode_id}", leave=False)):
        item = dataset[int(abs_idx)]
        batch = _batch_from_item(item, device)
        if pi_lang is not None:
            batch = pi_lang.inject_language_tokens(batch, device)

        with torch.no_grad():
            if infer_mode == "one_shot":
                chunk = policy.predict_action_chunk(batch)
                action = chunk[:, 0]
            else:
                action = policy.select_action(batch)

        states[i] = item["observation.state"].numpy()
        gt_actions[i] = item["actions"].numpy()
        pred_actions[i] = action[0].detach().float().cpu().numpy()

    return EpisodeResult(
        episode_index=episode_id,
        states=states,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        frame_indices=indices,
    )


def save_episode_npz(path: Path, result: EpisodeResult, infer_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        episode_index=result.episode_index,
        states=result.states,
        gt_actions=result.gt_actions,
        pred_actions=result.pred_actions,
        frame_indices=result.frame_indices,
        errors=result.errors,
        infer_mode=infer_mode,
        action_label=ACTION_LABEL,
    )


def load_episode_npz(path: Path) -> EpisodeResult:
    data = np.load(path, allow_pickle=False)
    return EpisodeResult(
        episode_index=int(data["episode_index"]),
        states=data["states"],
        gt_actions=data["gt_actions"],
        pred_actions=data["pred_actions"],
        frame_indices=data["frame_indices"],
    )


def plot_episode_overview(result: EpisodeResult, out_path: Path) -> None:
    """Three-row overview mirroring 1.replay_dataset.py + pred overlay."""
    gt = result.gt_actions
    pred = result.pred_actions
    states = result.states
    err = result.errors
    t = np.arange(len(gt))

    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for i in range(ARM_DIMS):
        axs[0].plot(t, gt[:, i], label=f"GT {ACTION_DIM_LABELS[i]}", alpha=0.9)
        axs[0].plot(
            t,
            pred[:, i],
            "--",
            label=f"Pred {ACTION_DIM_LABELS[i]}",
            alpha=0.75,
        )
    axs[0].set_title(
        f"Actions [{ACTION_LABEL}] GT vs Pred (Episode {result.episode_index})"
    )
    axs[0].set_ylabel("Value")
    axs[0].legend(loc="upper right", ncol=3, fontsize=8)
    axs[0].grid(True)

    axs[1].plot(t, gt[:, ARM_DIMS], label="GT gripper", color="C0")
    axs[1].plot(t, pred[:, ARM_DIMS], "--", label="Pred gripper", color="C1")
    axs[1].plot(t, states[:, ARM_DIMS], label="State gripper", alpha=0.6, color="C2")
    axs[1].set_title("Gripper: GT / Pred / State")
    axs[1].set_ylabel("Value")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    l2_arm = np.linalg.norm(err[:, :ARM_DIMS], axis=1)
    axs[2].plot(t, l2_arm, color="C3", label="|error| L2 (arm)")
    for i in range(ARM_DIMS):
        axs[2].plot(t, np.abs(err[:, i]), alpha=0.35, label=ACTION_DIM_LABELS[i])
    axs[2].set_title("Per-frame absolute error")
    axs[2].set_ylabel("|err|")
    axs[2].legend(loc="upper right", ncol=4, fontsize=8)
    axs[2].grid(True)

    gt_diff = np.diff(gt, axis=0)
    pred_diff = np.diff(pred, axis=0)
    for i in range(ARM_DIMS):
        axs[3].plot(
            t[1:], gt_diff[:, i], alpha=0.4, label=f"GT Δ{ACTION_DIM_LABELS[i]}"
        )
        axs[3].plot(
            t[1:],
            pred_diff[:, i],
            "--",
            alpha=0.4,
            label=f"Pred Δ{ACTION_DIM_LABELS[i]}",
        )
    axs[3].set_title("Action delta (smoothness)")
    axs[3].set_xlabel("Frame")
    axs[3].set_ylabel("Δ")
    axs[3].grid(True)

    m = result.summary()
    fig.suptitle(
        f"MAE={m['mae']:.5f}  RMSE={m['rmse']:.5f}  "
        f"arm_L2_mean={m['l2_arm_mean']:.5f}  frames={m['num_frames']}",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_episode_scatter(result: EpisodeResult, out_path: Path) -> None:
    gt = result.gt_actions
    pred = result.pred_actions
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.ravel()
    for i in range(ACTION_DIM):
        axs[i].scatter(gt[:, i], pred[:, i], s=8, alpha=0.5)
        lo = min(gt[:, i].min(), pred[:, i].min())
        hi = max(gt[:, i].max(), pred[:, i].max())
        axs[i].plot([lo, hi], [lo, hi], "r--", lw=1)
        axs[i].set_xlabel(f"GT {ACTION_DIM_LABELS[i]}")
        axs[i].set_ylabel(f"Pred {ACTION_DIM_LABELS[i]}")
        axs[i].set_title(ACTION_DIM_LABELS[i])
        axs[i].grid(True)
    axs[ACTION_DIM].axis("off")
    fig.suptitle(f"GT vs Pred scatter (Episode {result.episode_index})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_episode_error_hist(result: EpisodeResult, out_path: Path) -> None:
    err = result.errors
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.ravel()
    for i in range(ACTION_DIM):
        axs[i].hist(err[:, i], bins=40, alpha=0.85, color=f"C{i}")
        axs[i].axvline(0.0, color="k", lw=0.8)
        axs[i].set_title(f"{ACTION_DIM_LABELS[i]}  MAE={np.abs(err[:, i]).mean():.5f}")
        axs[i].set_xlabel("pred - gt")
        axs[i].grid(True)
    axs[ACTION_DIM].axis("off")
    fig.suptitle(f"Error distribution (Episode {result.episode_index})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global_summary(results: list[EpisodeResult], out_dir: Path) -> dict:
    if not results:
        return {}
    all_gt = np.concatenate([r.gt_actions for r in results], axis=0)
    all_pred = np.concatenate([r.pred_actions for r in results], axis=0)
    global_m = _metrics(all_pred, all_gt)

    per_ep_mae = np.array([r.summary()["mae"] for r in results])
    per_dim_mae = np.array([r.summary()["per_dim_mae"] for r in results])

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(ACTION_DIM)
    axs[0].bar(x - 0.15, global_m["per_dim_mae"], width=0.3, label="MAE")
    axs[0].bar(x + 0.15, global_m["per_dim_rmse"], width=0.3, label="RMSE")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(ACTION_DIM_LABELS, rotation=30)
    axs[0].set_title("Global per-dimension error")
    axs[0].legend()
    axs[0].grid(True, axis="y")

    axs[1].bar(np.arange(len(results)), per_ep_mae, color="C2")
    axs[1].set_xlabel("Episode index (eval order)")
    axs[1].set_ylabel("MAE")
    axs[1].set_title("Per-episode MAE")
    axs[1].set_xticks(np.arange(len(results)))
    axs[1].set_xticklabels([str(r.episode_index) for r in results], rotation=45)
    axs[1].grid(True, axis="y")

    im = axs[2].imshow(per_dim_mae.T, aspect="auto", cmap="magma")
    axs[2].set_yticks(np.arange(ACTION_DIM))
    axs[2].set_yticklabels(ACTION_DIM_LABELS)
    axs[2].set_xticks(np.arange(len(results)))
    axs[2].set_xticklabels([str(r.episode_index) for r in results], rotation=45)
    axs[2].set_title("Per-episode × per-dim MAE")
    fig.colorbar(im, ax=axs[2], fraction=0.046)

    fig.suptitle(
        f"Global MAE={global_m['mae']:.5f}  RMSE={global_m['rmse']:.5f}  "
        f"episodes={len(results)}  frames={len(all_gt)}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "global_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "action_label": ACTION_LABEL,
        "num_episodes": len(results),
        "num_frames": len(all_gt),
        "global": global_m,
        "episodes": [r.summary() for r in results],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_interactive(dataset: LeRobotDataset, result: EpisodeResult) -> None:
    """Slider UI: camera views + numeric state / GT / pred (like 1.replay_dataset.py)."""
    ep_id = result.episode_index
    n = len(result.frame_indices)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[5, 1.2, 0.5])
    ax_agent = fig.add_subplot(gs[0, 0])
    ax_wrist = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])
    ax_slider = fig.add_subplot(gs[2, :])
    ax_text.axis("off")

    init = dataset[int(result.frame_indices[0])]
    img_a = ax_agent.imshow(init["observation.image"].permute(1, 2, 0).numpy())
    img_w = ax_wrist.imshow(init["observation.wrist_image"].permute(1, 2, 0).numpy())
    ax_agent.set_title("Agent View")
    ax_agent.axis("off")
    ax_wrist.set_title("Wrist View")
    ax_wrist.axis("off")

    text_display = ax_text.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    slider = Slider(ax_slider, "Frame", 0, n - 1, valinit=0, valstep=1)

    def update(val: float) -> None:
        idx = int(slider.val)
        item = dataset[int(result.frame_indices[idx])]
        img_a.set_data(item["observation.image"].permute(1, 2, 0).numpy())
        img_w.set_data(item["observation.wrist_image"].permute(1, 2, 0).numpy())
        s = result.states[idx]
        gt = result.gt_actions[idx]
        pred = result.pred_actions[idx]
        err = result.errors[idx]
        text_display.set_text(
            f"FRAME: {idx:04d} | EPISODE: {ep_id}\n"
            f"STATE : {np.round(s, 4)}\n"
            f"GT    : {np.round(gt, 4)}\n"
            f"PRED  : {np.round(pred, 4)}\n"
            f"ERR   : {np.round(err, 4)}  |arm L2|={np.linalg.norm(err[:ARM_DIMS]):.5f}"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()


def parse_episodes(spec: str, num_episodes: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(num_episodes))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline dataset validation with plots"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="act",
        choices=("act", "pi0"),
        help="Policy backend",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./.ckpt/auboI10_act_w_2_view_temporal_ensemble_coeff09",
        help="Local checkpoint directory",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="0,1,2",
        help="Comma list or range (0-4) or 'all'",
    )
    parser.add_argument(
        "--infer-mode",
        type=str,
        default="select_action",
        choices=("select_action", "one_shot"),
        help="select_action: closed-loop queue; one_shot: predict_action_chunk[:,0]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./val_dataset_out",
        help="Directory for npz, png, summary.json",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES index",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open slider UI for the last evaluated episode",
    )
    parser.add_argument(
        "--plot-only",
        type=str,
        default=None,
        help="Skip inference; plot from saved episode npz (file or directory)",
    )
    parser.add_argument(
        "--no-interp",
        action="store_true",
        help="Use raw qpos dataset instead of interpolated copy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        src = Path(args.plot_only)
        npz_files = sorted(src.glob("episode_*.npz")) if src.is_dir() else [src]
        results = [load_episode_npz(p) for p in npz_files]
        for r in results:
            ep_dir = out_dir / f"episode_{r.episode_index:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            plot_episode_overview(r, ep_dir / "overview.png")
            plot_episode_scatter(r, ep_dir / "scatter.png")
            plot_episode_error_hist(r, ep_dir / "error_hist.png")
        plot_global_summary(results, out_dir)
        print(f"Replotted {len(results)} episodes -> {out_dir}")
        return

    MyPolicy.set_visible_cuda_devices(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = dataset_root(interp=not args.no_interp)
    print(f"ACTION_LABEL: {ACTION_LABEL}")
    print(f"Dataset: {root}")
    print(f"Policy: {args.policy}  checkpoint: {args.checkpoint}")
    print(f"Infer mode: {args.infer_mode}  device: {device}")

    dataset = LeRobotDataset(REPO_NAME, root=root)
    episode_ids = parse_episodes(args.episodes, dataset.num_episodes)
    print(f"Evaluating episodes: {episode_ids}")

    pi_lang: MyPIPolicy | None = None
    if args.policy == "act":
        policy = load_act_policy(args.checkpoint, device)
    else:
        policy, pi_lang = load_pi0_policy(args.checkpoint, device)

    results: list[EpisodeResult] = []
    for ep_id in episode_ids:
        result = infer_episode(
            policy,
            dataset,
            ep_id,
            device,
            pi_lang=pi_lang,
            infer_mode=args.infer_mode,
        )
        results.append(result)
        ep_dir = out_dir / f"episode_{ep_id:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        save_episode_npz(out_dir / f"episode_{ep_id:03d}.npz", result, args.infer_mode)
        plot_episode_overview(result, ep_dir / "overview.png")
        plot_episode_scatter(result, ep_dir / "scatter.png")
        plot_episode_error_hist(result, ep_dir / "error_hist.png")
        s = result.summary()
        print(
            f"  ep {ep_id:03d}: frames={s['num_frames']} "
            f"MAE={s['mae']:.6f} RMSE={s['rmse']:.6f} "
            f"arm_L2_mean={s['l2_arm_mean']:.6f}"
        )

    summary = plot_global_summary(results, out_dir)
    if summary:
        g = summary["global"]
        print("-" * 40)
        print(f"Global MAE={g['mae']:.6f}  RMSE={g['rmse']:.6f}")
        print(f"Saved to {out_dir}")

    if args.interactive and results:
        run_interactive(dataset, results[-1])


if __name__ == "__main__":
    main()
