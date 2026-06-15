#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate an openpi JAX checkpoint on Aubo MyEnv simulation via WebSocket.

Match policy config to action type:
  - pi0_auboI10_low_mem_finetune_qpos  ->  --action-type qpos
  - pi0_auboI10_low_mem_finetune_ee_pose -> --action-type ee_pose

Prerequisites
-------------
1. openpi policy server (separate terminal, openpi repo + .venv):

   cd /home/ningyu/code_before_paper/openpi
   CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
   uv run scripts/serve_policy.py \\
     --default-prompt "Put cube on the black platform" \\
     policy:checkpoint \\
     --policy.config=pi0_auboI10_low_mem_finetune_qpos \\
     --policy.dir=checkpoints/pi0_auboI10_low_mem_finetune_qpos/EXP/19999

2. openpi-client in this project's environment:

   uv pip install -e /home/ningyu/code_before_paper/openpi/packages/openpi-client

3. Run this script from MyI10Tele (MuJoCo / DISPLAY must work):

   cd /home/ningyu/MyI10Tele
   PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
     --action-type qpos --replan-steps 1 \\
     --dataset-init-episode 0 \\
     --lerobot-root ~/MyI10Tele/data_auboI10_qpos_v21_continuous

   Feed policy **exact training tensors** (pixels + state + task) while sim still executes actions::

   Sim closed-loop success-rate eval (teleop-tick ON by default; writes eval_summary.json)::

   PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
     --port 8000 --action-type qpos \\
     --action-delta-stride 10 --replan-steps 1 \\
     --policy-obs-source sim --num-episodes 20 \\
     --trace-dir ./openpi_eval_trace_qpos_sim

   Sim closed-loop (--policy-obs-source sim): policy sees live sim proprio. If arm
   drifts off the demo manifold, k10 AbsoluteActions amplify error (positive feedback).
   Hybrid A (state-dataset-image-sim) isolates proprio as the main failure mode.
   Fix: finetune with ``pi0_auboI10_low_mem_finetune_qpos_k10_state_noise`` (proprio
   noise σ=0.04 rad on arm dims during training).

   qpos + --action-delta-stride K: chunk[0] is absolute qpos at t+K (after server
   AbsoluteActions). Hold the same setpoint for K ticks (default). Optional
   --qpos-hold-ramp: open-loop anchor→target interpolation by hold index (not q_cur).

   Sim closed-loop anti-drift (delta qpos, stride=1 only)::

   # (1) Integrate chunk on live post: cmd[k]=sim+Δ[0], post+Δ[k]-Δ[k-1]; gripper absolute
   PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
     --action-type qpos --policy-obs-source sim --replan-steps 30 \\
     --qpos-chunk-integrate --trace-dir ./openpi_eval_integrate

   # (2) Receding horizon: infer every step, chunk[0] only (pair with --qpos-chunk-integrate)
   PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
     --action-type qpos --policy-obs-source sim \\
     --qpos-receding-horizon --qpos-chunk-integrate --trace-dir ./openpi_eval_receding

   Adaptive settle (stop when arm reaches target):

   PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
     --action-type qpos --action-delta-stride 10 \\
     --physics-settle-tol 0.005 --physics-settle-max-steps 200 \\
     --trace-dir ./openpi_eval_trace_qpos_k10

   GT open-loop replay (no policy server; same episode as eval_dataset ep0)::

   PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
     --action-type qpos --teleop-render \\
     --replay-gt-episode 0 \\
     --lerobot-root ~/MyI10Tele/data_auboI10_qpos_v21_continuous \\
     --physics-settle-steps 1 \\
     --trace-dir ./openpi_eval_trace_gt_ep0

4. Inspect traces:

   PYTHONPATH=src python src/core/8.val_openpi_sim.py --trace-dir ./openpi_eval_trace --trace-analyze-only

   Re-draw qpos time plots from saved traces::

   PYTHONPATH=src python src/core/8.val_openpi_sim.py --trace-dir ./openpi_eval_trace --trace-plot-only

   Sim grasp gap (dataset gripper not as tight as tight pair margin; XML unchanged)::

   PYTHONPATH=src python src/core/8.val_openpi_sim.py \\
     --action-type qpos --port 8000 \\
     --val-cube-gripper-margin 0.003 \\
     --val-gripper-close-boost 0.06 \\
     --trace-dir ./openpi_eval_trace_grasp_fix
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from core.dataset_config import (
    AUBOI10_QPOS_ROOT_V21_CORRECT,
    TASK_NAME,
    policy_ee_pose_command,
)
from core.eval_action_guard import (
    clamp_absolute_ee_action,
    cube_on_table,
    cube_pose,
)
from core.openpi_obs import (
    build_openpi_observation_from_lerobot_item,
    build_openpi_observation_from_sim,
    preprocess_lerobot_image,
)


XML_PATH = os.path.expanduser("~/MyI10Tele/assets/aubo_i10_inspire/myscene.xml")
HZ = 20
_QPOS_JOINT_LABELS = (
    "shoulder",
    "upperArm",
    "foreArm",
    "wrist1",
    "wrist2",
    "wrist3",
)
_EE_POSE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw")
_POLICY_OBS_USES_DATASET = frozenset(
    {"dataset", "state-dataset-image-sim", "state-sim-image-dataset"}
)


def _tele_observation_frame(
    env, prompt: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """One 20Hz tick: 0.tele.py order (pre_state → grab → resize) → policy dict."""
    pre_state = np.array(env.get_joint_state(), dtype=np.float32)
    agent_raw, wrist_raw = env.grab_image()
    element = build_openpi_observation_from_sim(agent_raw, wrist_raw, pre_state, prompt)
    return (
        pre_state,
        element["observation/image"],
        element["observation/wrist_image"],
        agent_raw,
        wrist_raw,
        element,
    )


class _DatasetPolicyObsStream:
    """LeRobot episode observations — same tensors training dataloader repacks for the model."""

    def __init__(
        self, lerobot_root: str, episode_id: int, repo_id: str = "auboI10"
    ) -> None:
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        root = os.path.expanduser(lerobot_root)
        self._meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        self._dataset = lerobot_dataset.LeRobotDataset(repo_id, root=root)
        self._indices = _episode_indices(self._dataset, episode_id)
        self._tasks = self._meta.tasks
        self.episode_id = episode_id

    def __len__(self) -> int:
        return len(self._indices)

    def obs_at(self, step: int, default_prompt: str) -> dict:
        if step >= len(self._indices):
            raise IndexError(
                f"step {step} >= dataset episode length {len(self._indices)}"
            )
        item = self._dataset[int(self._indices[step])]
        return build_openpi_observation_from_lerobot_item(
            item, self._tasks, default_prompt
        )


def _episode_indices(dataset, episode_id: int) -> np.ndarray:
    ep_table = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_id)
    return np.array(ep_table["index"], dtype=np.int64)


def _as_state(state) -> np.ndarray:
    return np.asarray(state, dtype=np.float32)


def _wrap_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi] (element-wise)."""
    return ((np.asarray(angles, dtype=np.float64) + np.pi) % (2 * np.pi)) - np.pi


def _pose_delta_6d(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """6D delta in the same space: xyz linear; rpy uses shortest angular difference."""
    d = np.asarray(target, dtype=np.float32)[:6] - np.asarray(ref, dtype=np.float32)[:6]
    d[3:6] = _wrap_pi(d[3:6]).astype(np.float32)
    return d


def _pose_delta_batch(targets: np.ndarray, refs: np.ndarray) -> np.ndarray:
    d = (
        np.asarray(targets, dtype=np.float32)[:, :6]
        - np.asarray(refs, dtype=np.float32)[:, :6]
    )
    d[:, 3:6] = _wrap_pi(d[:, 3:6]).astype(np.float32)
    return d


def _to_numpy7(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[-1] < 7:
        raise ValueError(f"Expected last dim >= 7, got shape {arr.shape}")
    return arr[:7].astype(np.float64, copy=False)


def _load_gt_episode(
    lerobot_root: str, episode_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load one episode from LeRobot v2 parquet (no LeRobotDataset — avoids v2/v3 API mismatch)."""
    import pandas as pd

    root = Path(os.path.expanduser(lerobot_root))
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset meta: {info_path}")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    num_episodes = int(info["total_episodes"])

    if episode_id < 0 or episode_id >= num_episodes:
        raise ValueError(
            f"episode_id {episode_id} out of range [0, {num_episodes - 1}]"
        )

    parquets = sorted((root / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet under {root / 'data'}")

    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    ep = df.loc[df["episode_index"] == episode_id].sort_values("frame_index")
    if ep.empty:
        raise ValueError(f"Episode {episode_id} has no rows in {root}")

    states = np.stack([_to_numpy7(v) for v in ep["observation.state"].values])
    actions = np.stack([_to_numpy7(v) for v in ep["actions"].values])
    obj_init = None
    if "obj_init" in ep.columns:
        obj_init = np.asarray(ep.iloc[0]["obj_init"], dtype=np.float64).reshape(-1)
    return states, actions, obj_init


def _snap_robot_to_state(env, state7: np.ndarray) -> None:
    """Set arm + gripper to a recorded 7D joint state (OpenPI layout)."""
    import mujoco

    from core.my_env import openpi_gripper_to_rh_r1_ctrl

    q_arm = np.asarray(state7[:6], dtype=np.float64)
    env.env.forward(q=q_arm, joint_names=env.joint_names, increase_tick=False)
    grip = openpi_gripper_to_rh_r1_ctrl(float(state7[6]))
    env.q = np.concatenate([q_arm, np.array([grip], dtype=np.float64)])
    env.compute_q = q_arm.copy()
    env.last_q = q_arm.copy()
    env.env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.env.model, env.env.data)


def _dataset_aligned_reset(
    env,
    *,
    seed: int,
    state0: np.ndarray,
    obj_init: np.ndarray | None,
    use_recorded_layout: bool,
    warmup_steps: int,
) -> None:
    """Reset and snap to dataset frame-0 state with teleop-consistent timing."""
    if use_recorded_layout and obj_init is not None:
        env.reset_with_recorded_layout(obj_init, seed=seed)
    else:
        env.reset(seed=seed)
    _snap_robot_to_state(env, state0)
    if warmup_steps < 0:
        raise ValueError(
            f"--dataset-init-warmup-steps must be >= 0, got {warmup_steps}"
        )
    for _ in range(warmup_steps):
        env.step_env()


def _apply_obj_init(env, obj_init: np.ndarray) -> None:
    env.apply_scene_layout(obj_init, settle=False)


class EpisodeTrace:
    """Per-episode rollout log for debugging policy I/O vs simulator state."""

    def __init__(
        self,
        episode_index: int,
        trace_dir: Path,
        ref_states: np.ndarray | None = None,
    ) -> None:
        self.episode_index = episode_index
        self.trace_dir = trace_dir
        self.ep_dir = trace_dir / f"episode_{episode_index:03d}"
        self.ep_dir.mkdir(parents=True, exist_ok=True)
        self.ref_states = ref_states

        self.step: list[int] = []
        # qpos_pre/post: policy-tick semantics (pre = fed to policy / logged cmd delta ref)
        self.qpos_pre: list[np.ndarray] = []
        self.qpos_post: list[np.ndarray] = []
        # measured sim proprio at tick start (always env.get_joint_state())
        self.sim_qpos_pre: list[np.ndarray] = []
        # ee_pose (7D xyz+rpy+gripper) available when action_type == "ee_pose"
        self.ee_pre: list[np.ndarray] = []
        self.ee_post: list[np.ndarray] = []
        self.action_executed: list[np.ndarray] = []
        # "cmd delta" depends on action_type (qpos vs ee_pose); computed in record_step.
        self.arm_delta_cmd: list[np.ndarray] = []
        self.gripper_cmd: list[float] = []
        self.gripper_state_pre: list[float] = []
        self.gripper_state_post: list[float] = []
        self.replan: list[bool] = []
        self.infer_ms: list[float] = []
        self.action_chunks: list[dict] = []

    def record_replan(
        self,
        step_idx: int,
        qpos_pre: np.ndarray,
        action_chunk: np.ndarray,
        infer_ms: float,
        *,
        action_type: str,
        ee_pre: np.ndarray | None = None,
    ) -> None:
        chunk6 = np.asarray(action_chunk, dtype=np.float32)[:, :6]
        if action_type == "qpos":
            arm_delta_chunk = (
                chunk6 - np.asarray(qpos_pre, dtype=np.float32)[:6]
            ).astype(np.float32)
        elif action_type == "ee_pose":
            if ee_pre is None:
                raise ValueError("ee_pre required when action_type=='ee_pose'")
            ref = np.asarray(ee_pre, dtype=np.float32)[:6]
            arm_delta_chunk = np.stack(
                [_pose_delta_6d(row, ref) for row in chunk6], axis=0
            )
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        row = {
            "step": int(step_idx),
            "qpos_pre": qpos_pre.astype(np.float32).tolist(),
            "action_chunk": action_chunk.astype(np.float32).tolist(),
            "infer_ms": float(infer_ms),
            "arm_delta_chunk": arm_delta_chunk.tolist(),
        }
        if ee_pre is not None:
            row["ee_pre"] = np.asarray(ee_pre, dtype=np.float32).tolist()
        self.action_chunks.append(row)

    def record_step(
        self,
        *,
        step_idx: int,
        qpos_pre: np.ndarray,
        qpos_post: np.ndarray,
        action_executed: np.ndarray,
        action_type: str,
        ee_pre: np.ndarray | None = None,
        ee_post: np.ndarray | None = None,
        replan: bool,
        infer_ms: float,
        sim_qpos_pre: np.ndarray | None = None,
    ) -> None:
        action_executed = np.asarray(action_executed, dtype=np.float32)
        qpos_pre = np.asarray(qpos_pre, dtype=np.float32)
        qpos_post = np.asarray(qpos_post, dtype=np.float32)
        if action_type == "qpos":
            arm_delta = action_executed[:6] - qpos_pre[:6]
        elif action_type == "ee_pose":
            if ee_pre is None or ee_post is None:
                raise ValueError("ee_pre/ee_post required when action_type=='ee_pose'")
            ee_pre = np.asarray(ee_pre, dtype=np.float32)
            ee_post = np.asarray(ee_post, dtype=np.float32)
            arm_delta = _pose_delta_6d(action_executed, ee_pre)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        self.step.append(int(step_idx))
        self.qpos_pre.append(qpos_pre)
        self.qpos_post.append(qpos_post)
        if sim_qpos_pre is not None:
            self.sim_qpos_pre.append(np.asarray(sim_qpos_pre, dtype=np.float32))
        if ee_pre is not None:
            self.ee_pre.append(np.asarray(ee_pre, dtype=np.float32))
        if ee_post is not None:
            self.ee_post.append(np.asarray(ee_post, dtype=np.float32))
        self.action_executed.append(action_executed)
        self.arm_delta_cmd.append(arm_delta)
        self.gripper_cmd.append(float(action_executed[6]))
        self.gripper_state_pre.append(float(qpos_pre[6]))
        self.gripper_state_post.append(float(qpos_post[6]))
        self.replan.append(bool(replan))
        self.infer_ms.append(float(infer_ms))

    def finalize(self, success: bool, num_steps: int, *, action_type: str) -> dict:
        if self.step:
            qpos_pre = np.stack(self.qpos_pre)
            qpos_post = np.stack(self.qpos_post)
            actions = np.stack(self.action_executed)
            arm_delta = np.stack(self.arm_delta_cmd)
            if action_type == "qpos":
                arm_track_err = np.linalg.norm(
                    actions[:, :6] - qpos_post[:, :6], axis=1
                )
                arm_range_ref = qpos_pre
            elif action_type == "ee_pose":
                ee_pre_st = (
                    np.stack(self.ee_pre)
                    if self.ee_pre
                    else np.zeros((0, 7), dtype=np.float32)
                )
                # Use control-cycle window metrics:
                # command at k is compared against ee_pre[k], and tracking against ee_pre[k+1]
                # (state at the next 20Hz control tick after command execution/settling).
                if len(ee_pre_st) >= 2:
                    arm_delta = _pose_delta_batch(actions[:-1], ee_pre_st[:-1])
                    arm_track_err = np.linalg.norm(
                        _pose_delta_batch(actions[:-1], ee_pre_st[1:]), axis=1
                    )
                    ee_motion_norm = np.linalg.norm(
                        _pose_delta_batch(ee_pre_st[1:], ee_pre_st[:-1]), axis=1
                    )
                    ee_xyz_cmd_norm = np.linalg.norm(arm_delta[:, :3], axis=1)
                    arm_range_ref = ee_pre_st[:-1]
                else:
                    arm_delta = np.zeros((0, 6), dtype=np.float32)
                    arm_track_err = np.zeros((0,), dtype=np.float32)
                    ee_motion_norm = np.zeros((0,), dtype=np.float32)
                    ee_xyz_cmd_norm = np.zeros((0,), dtype=np.float32)
                    arm_range_ref = np.zeros((0, 7), dtype=np.float32)
            else:
                raise ValueError(f"Unknown action_type: {action_type}")
            if action_type == "qpos":
                arm_cmd_norm = np.linalg.norm(arm_delta, axis=1)
                ee_motion_norm = ee_xyz_cmd_norm = None
            elif action_type == "ee_pose":
                arm_cmd_norm = np.linalg.norm(arm_delta, axis=1)
            else:
                raise ValueError(f"Unknown action_type: {action_type}")
        else:
            qpos_pre = qpos_post = actions = arm_delta = arm_track_err = (
                arm_cmd_norm
            ) = np.zeros((0,))
            arm_range_ref = np.zeros((0,))
            ee_motion_norm = ee_xyz_cmd_norm = None

        np.savez(
            self.ep_dir / "rollout.npz",
            step=np.asarray(self.step, dtype=np.int32),
            qpos_pre=qpos_pre,
            qpos_post=qpos_post,
            ee_pre=(
                np.stack(self.ee_pre)
                if self.ee_pre
                else np.zeros((0, 7), dtype=np.float32)
            ),
            ee_post=(
                np.stack(self.ee_post)
                if self.ee_post
                else np.zeros((0, 7), dtype=np.float32)
            ),
            sim_qpos_pre=(
                np.stack(self.sim_qpos_pre)
                if self.sim_qpos_pre
                else np.zeros((0, 7), dtype=np.float32)
            ),
            action_executed=actions,
            arm_delta_cmd=arm_delta,
            arm_track_err=arm_track_err,
            arm_cmd_norm=arm_cmd_norm,
            gripper_cmd=np.asarray(self.gripper_cmd, dtype=np.float32),
            gripper_state_pre=np.asarray(self.gripper_state_pre, dtype=np.float32),
            gripper_state_post=np.asarray(self.gripper_state_post, dtype=np.float32),
            replan=np.asarray(self.replan, dtype=bool),
            infer_ms=np.asarray(self.infer_ms, dtype=np.float32),
            ee_motion_norm=(
                ee_motion_norm
                if ee_motion_norm is not None
                else np.zeros((0,), dtype=np.float32)
            ),
            ee_xyz_cmd_norm=(
                ee_xyz_cmd_norm
                if ee_xyz_cmd_norm is not None
                else np.zeros((0,), dtype=np.float32)
            ),
            ref_states=(
                np.asarray(self.ref_states, dtype=np.float32)
                if self.ref_states is not None
                else np.zeros((0, 7), dtype=np.float32)
            ),
        )
        with open(self.ep_dir / "action_chunks.jsonl", "w", encoding="utf-8") as f:
            for row in self.action_chunks:
                f.write(json.dumps(row) + "\n")

        cmd_near_zero_thresh = 0.02 if action_type == "ee_pose" else 1e-4
        summary = {
            "episode_index": self.episode_index,
            "action_type": action_type,
            "success": success,
            "num_steps": num_steps,
            "num_replans": int(sum(self.replan)),
            "arm_cmd_norm_mean": (
                float(arm_cmd_norm.mean()) if len(arm_cmd_norm) else 0.0
            ),
            "arm_cmd_norm_max": float(arm_cmd_norm.max()) if len(arm_cmd_norm) else 0.0,
            "arm_cmd_near_zero_frac": (
                float((arm_cmd_norm < cmd_near_zero_thresh).mean())
                if len(arm_cmd_norm)
                else 1.0
            ),
            "arm_track_err_mean": (
                float(arm_track_err.mean()) if len(arm_track_err) else 0.0
            ),
            "arm_track_err_max": (
                float(arm_track_err.max()) if len(arm_track_err) else 0.0
            ),
            "gripper_toggles": (
                int(np.sum(np.abs(np.diff(np.asarray(self.gripper_cmd))) > 0.5))
                if len(self.gripper_cmd) > 1
                else 0
            ),
            "state_pre_range_arm": (
                np.ptp(arm_range_ref[:, :6], axis=0).tolist()
                if len(arm_range_ref)
                else []
            ),
            "action_range_arm": (
                np.ptp(actions[:, :6], axis=0).tolist() if len(actions) else []
            ),
        }
        if ee_xyz_cmd_norm is not None and len(ee_xyz_cmd_norm):
            summary["ee_xyz_cmd_norm_mean"] = float(ee_xyz_cmd_norm.mean())
            summary["ee_xyz_cmd_norm_max"] = float(ee_xyz_cmd_norm.max())
            summary["ee_xyz_near_zero_frac"] = float((ee_xyz_cmd_norm < 0.005).mean())
        if ee_motion_norm is not None and len(ee_motion_norm):
            summary["ee_motion_norm_mean"] = float(ee_motion_norm.mean())
        if self.ref_states is not None and self.sim_qpos_pre:
            sim = np.stack(self.sim_qpos_pre)
            n = min(len(sim), len(self.ref_states))
            ref = self.ref_states[:n]
            drift = np.linalg.norm(sim[:n, :6] - ref[:, :6], axis=1)
            summary["sim_state_drift_mean"] = float(drift.mean())
            summary["sim_state_drift_max"] = float(drift.max())
            summary["sim_state_drift_steps"] = int(n)
            for thresh in (0.02, 0.05, 0.10):
                key = f"sim_state_drift_first_over_{int(thresh * 100):03d}"
                summary[key] = (
                    int(np.argmax(drift > thresh)) if np.any(drift > thresh) else -1
                )
        qpos_plot = plot_trace_episode_qpos(
            self.ep_dir,
            trace_dir=self.trace_dir,
            action_type=action_type,
            ref_states=self.ref_states,
        )
        if qpos_plot is not None:
            summary["qpos_time_plot"] = str(qpos_plot)
        with open(self.ep_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def _resolve_qpos_plot_dir(trace_dir: Path) -> Path:
    return trace_dir / "qpos_time"


def plot_trace_episode_qpos(
    ep_dir: Path,
    *,
    trace_dir: Path,
    action_type: str,
    ref_states: np.ndarray | None = None,
) -> Path | None:
    """Plot qpos / command / ref vs control step from ``rollout.npz``."""
    import matplotlib.pyplot as plt

    npz_path = ep_dir / "rollout.npz"
    if not npz_path.is_file():
        return None

    data = np.load(npz_path)
    steps = np.asarray(data["step"], dtype=np.int32)
    if steps.size == 0:
        return None

    x = steps.astype(np.float64)
    qpos_post = np.asarray(data["qpos_post"], dtype=np.float32)
    actions = np.asarray(data["action_executed"], dtype=np.float32)
    sim_qpos = np.asarray(data["sim_qpos_pre"], dtype=np.float32)
    track_err = np.asarray(data["arm_track_err"], dtype=np.float32)
    replan = np.asarray(data["replan"], dtype=bool)

    if ref_states is None and "ref_states" in data:
        ref_arr = np.asarray(data["ref_states"], dtype=np.float32)
        ref_states = ref_arr if ref_arr.ndim == 2 and ref_arr.shape[0] else None

    has_sim = sim_qpos.ndim == 2 and sim_qpos.shape[0] == len(steps)
    n_ref = min(len(steps), len(ref_states)) if ref_states is not None else 0

    ee_pre = np.asarray(data["ee_pre"], dtype=np.float32)
    ee_post = np.asarray(data["ee_post"], dtype=np.float32)
    has_ee = ee_post.ndim == 2 and ee_post.shape[0] == len(steps)
    has_ee_pre = has_ee and ee_pre.ndim == 2 and ee_pre.shape[0] == len(steps)

    n_rows = 5 if has_ee else 3
    height_ratios = [1.0, 1.0, 0.85, 0.85, 0.85] if has_ee else [1.0, 1.0, 0.85]
    fig_h = 18 if has_ee else 12
    fig = plt.figure(figsize=(16, fig_h))
    gs = fig.add_gridspec(
        n_rows, 3, height_ratios=height_ratios, hspace=0.38, wspace=0.28
    )
    joint_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    replan_x = x[replan] if replan.any() else np.asarray([], dtype=np.float64)
    for i, ax in enumerate(joint_axes):
        ax.plot(x, qpos_post[:, i], label="post", color="C0")
        ax.plot(x, actions[:, i], "--", label="cmd", color="C1")
        if has_sim:
            ax.plot(x, sim_qpos[:, i], ":", label="sim_pre", color="C2", alpha=0.85)
        if n_ref:
            ax.plot(
                x[:n_ref],
                ref_states[:n_ref, i],
                "-.",
                label="ref",
                color="C3",
                alpha=0.85,
            )
        for rx in replan_x:
            ax.axvline(rx, color="gray", alpha=0.12, linewidth=0.8)
        ax.set_title(_QPOS_JOINT_LABELS[i])
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.35)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    ax_grip = fig.add_subplot(gs[2, 0])
    ax_grip.plot(x, qpos_post[:, 6], label="post", color="C0")
    ax_grip.plot(x, actions[:, 6], "--", label="cmd", color="C1")
    if has_sim:
        ax_grip.plot(x, sim_qpos[:, 6], ":", label="sim_pre", color="C2", alpha=0.85)
    if n_ref:
        ax_grip.plot(
            x[:n_ref],
            ref_states[:n_ref, 6],
            "-.",
            label="ref",
            color="C3",
            alpha=0.85,
        )
    ax_grip.set_title("gripper")
    ax_grip.set_ylabel("norm")
    ax_grip.legend(loc="upper right", fontsize=8)
    ax_grip.grid(True, alpha=0.35)

    ax_track = fig.add_subplot(gs[2, 1])
    # ee_pose: track_err[k] compares cmd at step k vs ee_pre at step k+1 → len N-1
    x_track = x[: len(track_err)]
    ax_track.plot(x_track, track_err, color="C4", label="|cmd-post| L2 arm")
    ax_track.set_title("arm tracking error")
    ax_track.set_ylabel("rad")
    ax_track.legend(loc="upper right", fontsize=8)
    ax_track.grid(True, alpha=0.35)

    qpost_diff = np.diff(qpos_post[:, :6], axis=0)
    ax_delta = fig.add_subplot(gs[2, 2])
    for i in range(6):
        ax_delta.plot(x[1:], qpost_diff[:, i], alpha=0.55, label=_QPOS_JOINT_LABELS[i])
    ax_delta.set_title("Δqpos_post (per tick)")
    ax_delta.set_ylabel("rad")
    ax_delta.legend(loc="upper right", fontsize=7, ncol=2)
    ax_delta.grid(True, alpha=0.35)

    bottom_axes = joint_axes + [ax_grip, ax_track, ax_delta]
    if has_ee:
        ee_axes = [fig.add_subplot(gs[3, c]) for c in range(3)] + [
            fig.add_subplot(gs[4, c]) for c in range(3)
        ]
        for i, ax in enumerate(ee_axes):
            dim = i if i < 6 else None
            if dim is not None:
                ax.plot(x, ee_post[:, dim], label="ee_post", color="C0")
                if has_ee_pre:
                    ax.plot(
                        x,
                        ee_pre[:, dim],
                        ":",
                        label="ee_pre",
                        color="C2",
                        alpha=0.85,
                    )
                if action_type == "ee_pose":
                    ax.plot(
                        x,
                        actions[:, dim],
                        "--",
                        label="ee_cmd",
                        color="C1",
                    )
                for rx in replan_x:
                    ax.axvline(rx, color="gray", alpha=0.12, linewidth=0.8)
                unit = "m" if dim < 3 else "rad"
                ax.set_title(f"ee {_EE_POSE_LABELS[dim]}")
                ax.set_ylabel(unit)
                ax.grid(True, alpha=0.35)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)
            else:
                if has_ee_pre:
                    ee_xyz_delta = ee_post[:, :3] - ee_pre[:, :3]
                else:
                    ee_xyz_delta = np.diff(ee_post[:, :3], axis=0)
                    x_delta = x[1:]
                if has_ee_pre:
                    x_delta = x
                ax.plot(
                    x_delta,
                    np.linalg.norm(ee_xyz_delta, axis=1),
                    color="C5",
                    label="|Δee_xyz|",
                )
                if has_ee_pre:
                    ax.plot(
                        x_delta,
                        ee_xyz_delta[:, 2],
                        color="C6",
                        alpha=0.75,
                        label="Δee_z",
                    )
                for rx in replan_x:
                    ax.axvline(rx, color="gray", alpha=0.12, linewidth=0.8)
                ax.set_title("ee motion (per tick)")
                ax.set_ylabel("m")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.35)
        bottom_axes = bottom_axes + ee_axes

    for ax in bottom_axes:
        ax.set_xlabel("Step")

    title = (
        f"qpos + ee_pose vs step [{action_type}] {ep_dir.name}  "
        f"steps={len(steps)}  replans={int(replan.sum())}"
        if has_ee
        else f"qpos vs step [{action_type}] {ep_dir.name}  "
        f"steps={len(steps)}  replans={int(replan.sum())}"
    )
    fig.suptitle(title, fontsize=11, y=1.01)
    plot_dir = _resolve_qpos_plot_dir(trace_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"{ep_dir.name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trace_dir_qpos(trace_dir: Path) -> list[Path]:
    """Re-plot qpos time figures under ``trace_dir/qpos_time/``."""
    paths: list[Path] = []
    for ep_dir in sorted(trace_dir.glob("episode_*")):
        summary_path = ep_dir / "summary.json"
        action_type = "qpos"
        if summary_path.is_file():
            with open(summary_path, encoding="utf-8") as f:
                action_type = json.load(f).get("action_type", action_type)
        out = plot_trace_episode_qpos(
            ep_dir, trace_dir=trace_dir, action_type=action_type
        )
        if out is not None:
            paths.append(out)
    return paths


def analyze_trace_dir(trace_dir: Path) -> None:
    summaries = sorted(trace_dir.glob("episode_*/summary.json"))
    if not summaries:
        print(f"No traces under {trace_dir}")
        return
    rows = []
    for p in summaries:
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    print(f"Trace analysis: {trace_dir} ({len(rows)} episodes)")
    print(
        "  arm_cmd_norm_mean (avg over ep):",
        np.mean([r["arm_cmd_norm_mean"] for r in rows]),
    )
    print(
        "  arm_cmd_near_zero_frac (avg):",
        np.mean([r["arm_cmd_near_zero_frac"] for r in rows]),
    )
    print(
        "  arm_track_err_mean (avg):",
        np.mean([r["arm_track_err_mean"] for r in rows]),
    )
    if any("sim_state_drift_mean" in r for r in rows):
        print(
            "  sim_state_drift_mean (avg):",
            np.mean([r.get("sim_state_drift_mean", 0.0) for r in rows]),
        )
        print(
            "  sim_state_drift_first_over_002 (avg step):",
            np.mean([r.get("sim_state_drift_first_over_002", -1) for r in rows]),
        )
    if any("ee_xyz_cmd_norm_mean" in r for r in rows):
        print(
            "  ee_xyz_cmd_norm_mean (avg):",
            np.mean([r.get("ee_xyz_cmd_norm_mean", 0.0) for r in rows]),
        )
        print(
            "  ee_motion_norm_mean (avg):",
            np.mean([r.get("ee_motion_norm_mean", 0.0) for r in rows]),
        )
    print("  successes:", sum(r["success"] for r in rows), "/", len(rows))
    _print_failure_breakdown(rows)
    print("\nPer-episode:")
    for r in rows:
        extra = ""
        if "ee_xyz_cmd_norm_mean" in r:
            extra = (
                f" xyz_cmd={r['ee_xyz_cmd_norm_mean']:.5f}"
                f" ee_motion={r.get('ee_motion_norm_mean', 0.0):.5f}"
            )
        outcome = r.get("outcome", "success" if r["success"] else "?")
        failed = r.get("failed_criteria", [])
        fail_msg = "" if r["success"] else f" outcome={outcome} fail={failed}"
        print(
            f"  ep {r['episode_index']:03d}: steps={r['num_steps']} "
            f"cmd_norm={r['arm_cmd_norm_mean']:.5f} near_zero={r['arm_cmd_near_zero_frac']:.2%} "
            f"track_err={r['arm_track_err_mean']:.5f}{extra} "
            f"grip_toggles={r['gripper_toggles']} ok={r['success']}{fail_msg}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="openpi WebSocket eval on Aubo MyEnv sim"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="openpi serve_policy host"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="openpi serve_policy port"
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=1,
        help="Actions per chunk before re-infer (stride=1 configs only); ignored when --action-delta-stride>1",
    )
    parser.add_argument(
        "--action-delta-stride",
        type=int,
        default=1,
        metavar="K",
        help="Plan C / k10: re-infer every K sim steps; hold chunk[0] (q_{t+K}) for K ticks. Match train action_delta_stride.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=TASK_NAME,
        help="Language instruction for the policy",
    )
    parser.add_argument(
        "--xml-path", type=str, default=XML_PATH, help="MuJoCo scene XML"
    )
    parser.add_argument("--seed", type=int, default=42, help="MyEnv RNG seed")
    parser.add_argument(
        "--num-episodes", type=int, default=10, help="Number of evaluation rollouts"
    )
    parser.add_argument(
        "--max-steps", type=int, default=600, help="Max control steps per episode"
    )
    parser.add_argument(
        "--physics-settle-steps",
        type=int,
        default=50,
        metavar="N",
        help="Fixed mj_step count after each step(action) on GT replay path. "
        "Policy eval uses --teleop-tick (default) instead. "
        "Ignored when --physics-settle-tol is set.",
    )
    parser.add_argument(
        "--teleop-tick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Policy eval default ON: one mj_step after step(action) (0.tele.py L143-144). "
        "Matches dataset state/action timing. GT replay ignores this (uses --physics-settle-steps). "
        "Use --no-teleop-tick only for ablation.",
    )
    parser.add_argument(
        "--physics-settle-tol",
        type=float,
        default=0.0005,
        metavar="RAD",
        help="Adaptive settle: mj_step until ||q_arm - q_target|| <= RAD (cap: --physics-settle-max-steps)",
    )
    parser.add_argument(
        "--physics-settle-max-steps",
        type=int,
        default=1000,
        metavar="N",
        help="Max mj_step budget when --physics-settle-tol is set",
    )
    parser.add_argument(
        "--post-action-wait-s",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Policy eval: after each step(action), hold ctrl and mj_step for SEC simulation "
        "seconds, then re-observe and re-infer (one action per cycle). 0=use teleop-tick/settle.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Rollout videos. Default: <trace-dir>/videos when --trace-dir is set, "
        "else ./episode_videos_openpi",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default="qpos",
        choices=("qpos", "ee_pose"),
        help="qpos: joint targets; ee_pose: absolute flange xyz+rpy+gripper (match train config)",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="If set, save per-episode npz/json traces (state, actions, chunks)",
    )
    parser.add_argument(
        "--trace-episodes",
        type=int,
        default=0,
        help="Number of episodes to trace (from the start); 0 = trace all",
    )
    parser.add_argument(
        "--trace-analyze-only",
        action="store_true",
        help="Print trace summary from --trace-dir and exit",
    )
    parser.add_argument(
        "--trace-plot-only",
        action="store_true",
        help="Re-draw qpos plots under <trace-dir>/qpos_time/ from rollout.npz and exit",
    )
    parser.add_argument(
        "--teleop-render",
        action="store_true",
        help="Match 0.tele.py viewer overlays (camera panels + key hints)",
    )
    parser.add_argument(
        "--no-ee-guard",
        action="store_true",
        help="Disable per-step EE clamp (ee_pose only; default is guard ON)",
    )
    parser.add_argument(
        "--max-ee-xyz-step",
        type=float,
        default=0.005,
        help="Max |Δxyz| per control step when EE guard is on (meters; ~teleop dpos)",
    )
    parser.add_argument(
        "--max-ee-rpy-step",
        type=float,
        default=0.08,
        help="Max |Δrpy| per control step when EE guard is on (radians)",
    )
    parser.add_argument(
        "--log-cube-every",
        type=int,
        default=25,
        help="Print cube xyz every N steps; 0 = only when cube leaves the table",
    )
    parser.add_argument(
        "--replay-gt-episode",
        type=int,
        default=None,
        metavar="EP",
        help="Open-loop replay recorded actions for this dataset episode (no policy server)",
    )
    parser.add_argument(
        "--lerobot-root",
        type=str,
        default=AUBOI10_QPOS_ROOT_V21_CORRECT,
        help="LeRobot root for GT replay / --dataset-init-episode",
    )
    parser.add_argument(
        "--dataset-init-episode",
        type=int,
        default=None,
        metavar="EP",
        help="Restore obj_init + frame-0 qpos from this dataset episode each rollout (train/eval layout match)",
    )
    parser.add_argument(
        "--dataset-init-warmup-steps",
        type=int,
        default=20,
        metavar="N",
        help="After snapping to dataset frame-0 qpos, run N step_env() ticks. "
        "Default 0 matches recording pre_state timing.",
    )
    parser.add_argument(
        "--policy-obs-source",
        type=str,
        default="sim",
        choices=(
            "sim",
            "dataset",
            "state-dataset-image-sim",
            "state-sim-image-dataset",
        ),
        help="sim / dataset (both modalities same source); hybrid: cross GT vs sim for drift A/B",
    )
    parser.add_argument(
        "--policy-obs-dataset-episode",
        type=int,
        default=0,
        metavar="EP",
        help="LeRobot episode for --policy-obs-source=dataset (must match --lerobot-root / train data)",
    )
    parser.add_argument(
        "--policy-obs-state-alpha",
        type=float,
        default=0.0,
        metavar="A",
        help="sim closed-loop only: policy state = (1-A)*sim_qpos + A*demo_qpos[step] "
        "(needs --dataset-init-episode / ref_states). 0=pure sim, 1=Hybrid A state only.",
    )
    parser.add_argument(
        "--replay-max-frames",
        type=int,
        default=None,
        help="Cap GT replay length (default: full episode)",
    )
    parser.add_argument(
        "--no-replay-dataset-init",
        action="store_true",
        help="After reset, do not restore cube/platform from dataset obj_init",
    )
    parser.add_argument(
        "--arm-kp",
        type=float,
        default=None,
        help="Advanced: override XML arm position kp (default: keep aubo_i10_inspire.xml)",
    )
    parser.add_argument(
        "--arm-kv",
        type=float,
        default=None,
        help="Advanced: override XML arm position kv (default: keep aubo_i10_inspire.xml)",
    )
    parser.add_argument(
        "--gripper-kp",
        type=float,
        default=None,
        help="Override rh_r1_servo position kp (default: keep aubo_i10_inspire.xml)",
    )
    parser.add_argument(
        "--gripper-kv",
        type=float,
        default=None,
        help="Override rh_r1_servo position kv (default: keep aubo_i10_inspire.xml)",
    )
    parser.add_argument(
        "--gripper-forcerange",
        type=float,
        default=None,
        help="Symmetric force cap (N) on rh_r1_servo; raise if kp alone cannot close fast (try 300–500)",
    )
    parser.add_argument(
        "--qpos-hold-ramp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="k10 hold: linear setpoint anchor→chunk[0] by hold index (default: constant chunk[0])",
    )
    parser.add_argument(
        "--qpos-chunk-integrate",
        action="store_true",
        help="Requires --qpos-receding-horizon: cmd_arm = sim + (chunk[0]-infer) each step; "
        "gripper absolute. Ignored when replan_steps>1 per infer.",
    )
    parser.add_argument(
        "--qpos-receding-horizon",
        action="store_true",
        help="qpos stride=1: infer every step (chunk[0] only). Pair with --qpos-chunk-integrate for tight closed-loop",
    )
    parser.add_argument(
        "--qpos-exec-via-ik",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental: capped EE delta toward FK(q_policy) then IK. "
        "Only with stride=1; incompatible with hold ramp.",
    )
    parser.add_argument(
        "--val-cube-gripper-margin",
        type=float,
        default=None,
        metavar="M",
        help="Val-only runtime patch: cube↔gripper MuJoCo pair margin (m). "
        "Does not edit XML. Try 0.0025–0.004 when dataset gripper closure is looser than sim needs.",
    )
    parser.add_argument(
        "--val-gripper-close-boost",
        type=float,
        default=0.0,
        metavar="D",
        help="Val-only: add D to gripper command when action[6]>0.5 before sim step (e.g. 0.06).",
    )
    return parser.parse_args()


def _log_xml_arm_gains(env) -> None:
    kp, kv = env.get_arm_position_gains()
    grip_kp, grip_kv = env.get_gripper_position_gains()
    print(f"Arm actuators (aubo_i10_inspire.xml): kp={kp} kv={kv}")
    print(f"Gripper actuator (rh_r1_servo): kp={grip_kp} kv={grip_kv}")


_GRIPPER_COL_GEOM_NAMES = frozenset(
    {
        "gripper_base_col",
        "gripper_r1_col",
        "gripper_r2_col",
        "gripper_l1_col",
        "gripper_l2_col",
    }
)


def _apply_val_gripper_boost(
    action: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    boost = float(args.val_gripper_close_boost)
    if boost == 0.0:
        return action
    out = np.asarray(action, dtype=np.float64).copy()
    if out[6] > 0.5:
        out[6] = float(np.clip(out[6] + boost, 0.0, 1.0))
    return out


def _qpos_hold_setpoint(
    q_anchor: np.ndarray,
    q_target: np.ndarray,
    hold_stride: int,
    hold_step: int,
) -> np.ndarray:
    """Open-loop setpoint for tick hold_step in [0, K-1] within one replan window."""
    q_anchor = np.asarray(q_anchor, dtype=np.float64)
    q_target = np.asarray(q_target, dtype=np.float64)
    if hold_stride < 2:
        return q_target.copy()
    hold_step = int(hold_step)
    if hold_step < 0 or hold_step >= hold_stride:
        raise ValueError(f"hold_step {hold_step} out of range [0, {hold_stride - 1}]")
    alpha = float(hold_step + 1) / float(hold_stride)
    return q_anchor + alpha * (q_target - q_anchor)


def _qpos_chunk_horizon(args: argparse.Namespace, chunk_len: int) -> int:
    if args.qpos_receding_horizon:
        return 1
    return min(args.replan_steps, chunk_len)


def _qpos_chunk_integrated_cmd(
    chunk: np.ndarray,
    chunk_idx: int,
    infer_state_pre: np.ndarray,
    sim_qpos_pre: np.ndarray,
) -> np.ndarray:
    """Receding k=0 only: cmd_arm = sim + (chunk[0]-infer) = sim + Δ[0]. Gripper absolute."""
    if chunk_idx != 0:
        raise ValueError("chunk integrate applies only to chunk[0] (receding horizon)")
    chunk = np.asarray(chunk, dtype=np.float64)
    infer_state_pre = np.asarray(infer_state_pre, dtype=np.float64)
    sim_qpos_pre = np.asarray(sim_qpos_pre, dtype=np.float64)
    out = chunk[0, :7].copy()
    out[:6] = sim_qpos_pre[:6] + (chunk[0, :6] - infer_state_pre[:6])
    return out


def _blend_policy_state(
    sim_state: np.ndarray,
    ref_state: np.ndarray,
    alpha: float,
) -> np.ndarray:
    a = float(alpha)
    sim_state = np.asarray(sim_state, dtype=np.float32)
    ref_state = np.asarray(ref_state, dtype=np.float32)
    return ((1.0 - a) * sim_state + a * ref_state).astype(np.float32)


def _apply_eval_runtime_options(args: argparse.Namespace) -> None:
    if args.post_action_wait_s > 0.0:
        if args.teleop_tick:
            args.teleop_tick = False
            print(
                "NOTE: --post-action-wait-s → --no-teleop-tick "
                "(wait replaces per-tick settle)"
            )
        if args.replan_steps != 1:
            print(
                f"NOTE: --post-action-wait-s forces infer every step "
                f"(--replan-steps {args.replan_steps} ignored)"
            )
        if args.action_delta_stride > 1:
            raise ValueError("--post-action-wait-s requires --action-delta-stride 1")
    if "--physics-settle-tol" in sys.argv and args.teleop_tick:
        args.teleop_tick = False
        print(
            "NOTE: --physics-settle-tol on CLI → --no-teleop-tick "
            "(adaptive settle; teleop-tick would ignore tol)"
        )
    if args.physics_settle_tol > 0.05:
        print(
            f"WARN: --physics-settle-tol={args.physics_settle_tol} rad is very loose "
            f"(~{np.degrees(args.physics_settle_tol):.0f}°); try 0.005 for arm tracking"
        )
    if args.qpos_chunk_integrate and not args.qpos_receding_horizon:
        print(
            "WARN: --qpos-chunk-integrate without --qpos-receding-horizon has no effect "
            "(k>0 must use absolute chunk[k]; re-anchor was wrong for k>=1)"
        )


def _validate_qpos_closed_loop_exec(args: argparse.Namespace) -> None:
    if not (args.qpos_chunk_integrate or args.qpos_receding_horizon):
        return
    if args.action_type != "qpos":
        raise ValueError(
            "--qpos-chunk-integrate / --qpos-receding-horizon require --action-type qpos"
        )
    if args.action_delta_stride > 1:
        raise ValueError(
            "--qpos-chunk-integrate / --qpos-receding-horizon require --action-delta-stride 1"
        )
    if args.qpos_chunk_integrate and args.qpos_exec_via_ik:
        raise ValueError(
            "--qpos-chunk-integrate is incompatible with --qpos-exec-via-ik"
        )
    if args.qpos_chunk_integrate and not args.qpos_receding_horizon:
        return
    if args.qpos_receding_horizon and not args.qpos_chunk_integrate:
        print(
            "NOTE: --qpos-receding-horizon without --qpos-chunk-integrate "
            "uses absolute chunk[0] from server"
        )


def _physics_settle_desc(args: argparse.Namespace) -> str:
    if args.post_action_wait_s > 0.0:
        return f"post-action-wait={args.post_action_wait_s}s (sim time)"
    if args.teleop_tick:
        return "teleop-tick (1 mj_step)"
    if args.physics_settle_tol is not None:
        return (
            f"tol={args.physics_settle_tol} max_steps={args.physics_settle_max_steps}"
        )
    return f"steps={args.physics_settle_steps}"


def _physics_settle(env, args: argparse.Namespace) -> tuple[int, float]:
    if args.post_action_wait_s > 0.0:
        n = env.wait_physics(args.post_action_wait_s)
        target = np.asarray(env.compute_q[:6], dtype=np.float64)
        q = np.asarray(
            env.env.get_qpos_joints(joint_names=env.joint_names), dtype=np.float64
        )
        return n, float(np.linalg.norm(q - target))
    if args.teleop_tick:
        env.step_env()
        target = np.asarray(env.compute_q[:6], dtype=np.float64)
        q = np.asarray(
            env.env.get_qpos_joints(joint_names=env.joint_names), dtype=np.float64
        )
        return 1, float(np.linalg.norm(q - target))
    if args.physics_settle_tol is not None:
        return env.settle_physics(
            tol_rad=float(args.physics_settle_tol),
            max_steps=int(args.physics_settle_max_steps),
        )
    return env.settle_physics(steps=int(args.physics_settle_steps))


class _PolicyObsPack:
    __slots__ = ("state_pre", "agent_raw", "wrist_raw", "element")

    def __init__(
        self,
        state_pre: np.ndarray,
        agent_raw: np.ndarray,
        wrist_raw: np.ndarray,
        element: dict,
    ) -> None:
        self.state_pre = state_pre
        self.agent_raw = agent_raw
        self.wrist_raw = wrist_raw
        self.element = element


def _build_policy_observation(
    env,
    args: argparse.Namespace,
    step: int,
    dataset_obs: _DatasetPolicyObsStream | None,
    ref_states: np.ndarray | None,
) -> _PolicyObsPack | None:
    if args.policy_obs_source == "dataset":
        if dataset_obs is None or step >= len(dataset_obs):
            return None
        element = dataset_obs.obs_at(step, args.prompt)
        state_pre = _as_state(element["observation/state"])
        agent_raw, wrist_raw = env.grab_image()
        return _PolicyObsPack(state_pre, agent_raw, wrist_raw, element)
    if args.policy_obs_source == "state-dataset-image-sim":
        if dataset_obs is None or step >= len(dataset_obs):
            return None
        _, _, _, agent_raw, wrist_raw, _ = _tele_observation_frame(env, args.prompt)
        ds_element = dataset_obs.obs_at(step, args.prompt)
        state_pre = _as_state(ds_element["observation/state"])
        element = {
            "observation/image": ds_element["observation/image"],
            "observation/wrist_image": ds_element["observation/wrist_image"],
            "observation/state": state_pre,
            "prompt": args.prompt,
        }
        return _PolicyObsPack(state_pre, agent_raw, wrist_raw, element)
    if args.policy_obs_source == "state-sim-image-dataset":
        if dataset_obs is None or step >= len(dataset_obs):
            return None
        state_pre, _, _, agent_raw, wrist_raw, sim_element = _tele_observation_frame(
            env, args.prompt
        )
        ds_element = dataset_obs.obs_at(step, args.prompt)
        element = {
            "observation/image": ds_element["observation/image"],
            "observation/wrist_image": ds_element["observation/wrist_image"],
            "observation/state": sim_element["observation/state"],
            "prompt": args.prompt,
        }
        return _PolicyObsPack(state_pre, agent_raw, wrist_raw, element)
    state_pre, _, _, agent_raw, wrist_raw, element = _tele_observation_frame(
        env, args.prompt
    )
    if (
        args.policy_obs_state_alpha > 0.0
        and ref_states is not None
        and step < len(ref_states)
    ):
        state_pre = _blend_policy_state(
            state_pre, ref_states[step], args.policy_obs_state_alpha
        )
        element["observation/state"] = state_pre
    return _PolicyObsPack(state_pre, agent_raw, wrist_raw, element)


def _call_policy_infer(
    client,
    element: dict,
    *,
    episode_idx: int,
    step: int,
    action_type: str,
    ee_pre: np.ndarray | None,
    state_pre: np.ndarray,
    qpos_integrate: bool,
) -> tuple[np.ndarray, float]:
    print(
        f"[ep {episode_idx} step {step}] Calling policy infer "
        f"(first call may take 1-10 min for JAX compile)...",
        flush=True,
    )
    infer_out = client.infer(element)
    infer_ms = float(infer_out.get("policy_timing", {}).get("infer_ms", 0.0))
    chunk = np.asarray(infer_out["actions"], dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[np.newaxis, :]
    if chunk.shape[-1] < 7:
        raise ValueError(
            f"Expected actions with last dim >= 7, got shape {chunk.shape}"
        )
    if action_type == "ee_pose" and ee_pre is not None:
        delta0 = _pose_delta_6d(chunk[0], ee_pre)
        delta_msg = f"ee_delta[0]={delta0} |xyz|={np.linalg.norm(delta0[:3]):.4f}"
    elif qpos_integrate and action_type == "qpos":
        delta0 = chunk[0, :6] - state_pre[:6]
        delta_msg = f"qpos_integrate Δ[0]={delta0} |Δ|={np.linalg.norm(delta0):.4f}"
    else:
        delta_msg = f"qpos_delta[0]={(chunk[0, :6] - state_pre[:6])}"
    print(
        f"[ep {episode_idx} step {step}] infer done: "
        f"chunk={chunk.shape} infer_ms={infer_ms:.0f} {delta_msg}",
        flush=True,
    )
    return chunk, infer_ms


def _action_from_chunk(
    chunk: np.ndarray,
    infer_state: np.ndarray,
    sim_qpos: np.ndarray,
    args: argparse.Namespace,
    ee_pre: np.ndarray | None,
    *,
    ee_guard: bool,
) -> np.ndarray:
    action_np = np.asarray(chunk[0, :7], dtype=np.float64)
    if (
        args.qpos_chunk_integrate
        and args.qpos_receding_horizon
        and args.action_type == "qpos"
    ):
        action_np = _qpos_chunk_integrated_cmd(chunk, 0, infer_state, sim_qpos)
    if ee_guard:
        if ee_pre is None:
            raise RuntimeError("ee_pre required when EE guard is on")
        action_np, _ = clamp_absolute_ee_action(
            action_np,
            ee_pre,
            max_xyz_step=args.max_ee_xyz_step,
            max_rpy_step=args.max_ee_rpy_step,
        )
    return _apply_val_gripper_boost(action_np, args)


def _run_wait_replan_episode(
    env,
    client,
    args: argparse.Namespace,
    episode: int,
    *,
    dataset_obs: _DatasetPolicyObsStream | None,
    ref_states: np.ndarray | None,
    trace: EpisodeTrace | None,
    video_recorder,
    ee_guard: bool,
) -> tuple[bool, int, bool]:
    """act → wait → observe → infer each cycle."""
    ep_idx = episode + 1
    step = 0
    episode_success = False
    cube_fell_logged = False
    qpos_integrate = bool(
        args.qpos_chunk_integrate
        and args.qpos_receding_horizon
        and args.action_type == "qpos"
    )

    obs0 = _build_policy_observation(env, args, step, dataset_obs, ref_states)
    if obs0 is None:
        print(f"Episode {ep_idx}: no policy observation at step 0")
        return False, 0, False
    video_recorder.record_frame(obs0.agent_raw, obs0.wrist_raw)

    ee_pre = _as_state(env.get_ee_pose())
    chunk, infer_ms = _call_policy_infer(
        client,
        obs0.element,
        episode_idx=ep_idx,
        step=step,
        action_type=args.action_type,
        ee_pre=ee_pre,
        state_pre=obs0.state_pre,
        qpos_integrate=qpos_integrate,
    )
    if trace is not None:
        trace.record_replan(
            step,
            obs0.state_pre,
            chunk,
            infer_ms,
            action_type=args.action_type,
            ee_pre=ee_pre,
        )
    action_np = _action_from_chunk(
        chunk,
        obs0.state_pre,
        obs0.state_pre,
        args,
        ee_pre,
        ee_guard=ee_guard,
    )

    while env.env.is_viewer_alive() and step < args.max_steps:
        sim_qpos_pre = np.array(env.get_joint_state(), dtype=np.float32)
        if (
            args.policy_obs_source == "sim"
            and ref_states is not None
            and step < len(ref_states)
            and step % 10 == 0
        ):
            drift = float(np.linalg.norm(sim_qpos_pre[:6] - ref_states[step, :6]))
            print(
                f"[ep {ep_idx} step {step}] sim-vs-demo drift={drift:.4f} rad",
                flush=True,
            )

        env.step(action_np)
        _physics_settle(env, args)
        qpos_post = _as_state(env.get_joint_state())
        ee_post = _as_state(env.get_ee_pose())

        obs = _build_policy_observation(env, args, step, dataset_obs, ref_states)
        if obs is None:
            print(f"Episode {ep_idx}: dataset obs exhausted at step {step}")
            break
        video_recorder.record_frame(obs.agent_raw, obs.wrist_raw)
        ee_pre = _as_state(env.get_ee_pose())

        chunk, infer_ms = _call_policy_infer(
            client,
            obs.element,
            episode_idx=ep_idx,
            step=step,
            action_type=args.action_type,
            ee_pre=ee_pre,
            state_pre=obs.state_pre,
            qpos_integrate=qpos_integrate,
        )
        if trace is not None:
            trace.record_replan(
                step,
                obs.state_pre,
                chunk,
                infer_ms,
                action_type=args.action_type,
                ee_pre=ee_pre,
            )
            trace.record_step(
                step_idx=step,
                qpos_pre=sim_qpos_pre,
                qpos_post=qpos_post,
                action_executed=action_np,
                action_type=args.action_type,
                ee_pre=ee_pre,
                ee_post=ee_post,
                replan=True,
                infer_ms=infer_ms,
                sim_qpos_pre=sim_qpos_pre,
            )

        action_np = _action_from_chunk(
            chunk,
            obs.state_pre,
            obs.state_pre,
            args,
            ee_pre,
            ee_guard=ee_guard,
        )
        env.render(teleop=args.teleop_render)
        step += 1

        p_cube = cube_pose(env)
        on_table = cube_on_table(p_cube)
        if args.log_cube_every > 0 and step % args.log_cube_every == 0:
            print(
                f"[ep {ep_idx} step {step}] cube xyz={p_cube.round(4)} on_table={on_table}",
                flush=True,
            )
        if not on_table and not cube_fell_logged:
            cube_fell_logged = True
            print(
                f"[ep {ep_idx} step {step}] cube left table: xyz={p_cube.round(4)}",
                flush=True,
            )
        if env.check_success():
            print(f"Episode {ep_idx}: success in {step} steps")
            episode_success = True
            break

    return episode_success, step, cube_fell_logged


def _serialize_success_criteria(c: dict) -> dict:
    return {
        "success": bool(c["success"]),
        "xy_ok": bool(c["xy_ok"]),
        "z_ok": bool(c["z_ok"]),
        "gripper_open": bool(c["gripper_open"]),
        "ee_away": bool(c["ee_away"]),
        "xy_dist_m": float(c["xy_dist_m"]),
        "place_tol_xy_m": float(c["place_tol_xy_m"]),
        "cube_z_m": float(c["cube_z_m"]),
        "z_min_m": float(c["z_min_m"]),
        "rh_r1": float(c["rh_r1"]),
        "gripper_open_thresh": float(c["gripper_open_thresh"]),
        "ee_z_above_cube_m": float(c["ee_z_above_cube_m"]),
        "ee_away_thresh_m": float(c["ee_away_thresh_m"]),
        "cube_xyz": np.asarray(c["cube_xyz"], dtype=np.float64).tolist(),
        "platform_xyz": np.asarray(c["platform_xyz"], dtype=np.float64).tolist(),
        "ee_xyz": np.asarray(c["ee_xyz"], dtype=np.float64).tolist(),
    }


def _episode_outcome(
    env,
    *,
    success: bool,
    num_steps: int,
    max_steps: int,
    cube_left_table: bool,
) -> dict:
    """Structured end-of-episode outcome for success-rate statistics."""
    c = env.success_criteria()
    failed: list[str] = []
    if not c["xy_ok"]:
        failed.append("xy_off_platform")
    if not c["z_ok"]:
        failed.append("cube_below_deck")
    if not c["gripper_open"]:
        failed.append("gripper_not_open")
    if not c["ee_away"]:
        failed.append("ee_too_close_to_cube")

    timeout = not success and num_steps >= max_steps
    if success:
        primary = "success"
    elif timeout:
        primary = "timeout:" + (failed[0] if failed else "criteria_unmet")
    elif failed:
        primary = failed[0]
    else:
        primary = "unknown"

    return {
        "outcome": primary,
        "failed_criteria": failed,
        "timeout": bool(timeout),
        "cube_left_table": bool(cube_left_table),
        "end_criteria": _serialize_success_criteria(c),
    }


def _aggregate_failure_stats(episodes: list[dict]) -> dict:
    stats: dict[str, int] = {"success": 0, "timeout": 0, "cube_left_table": 0}
    for ep in episodes:
        if ep.get("success"):
            stats["success"] += 1
        if ep.get("timeout"):
            stats["timeout"] += 1
        if ep.get("cube_left_table"):
            stats["cube_left_table"] += 1
        if not ep.get("success"):
            for key in ep.get("failed_criteria", []):
                stats[key] = stats.get(key, 0) + 1
    return stats


def _print_failure_breakdown(episodes: list[dict]) -> None:
    stats = _aggregate_failure_stats(episodes)
    n = len(episodes)
    if n == 0:
        return
    labels = {
        "success": "成功",
        "timeout": "超时(max_steps)",
        "cube_left_table": "方块离桌(过程中)",
        "xy_off_platform": "方块XY未进平台",
        "cube_below_deck": "方块高度不足",
        "gripper_not_open": "夹爪未张开",
        "ee_too_close_to_cube": "EE离方块太近",
    }
    print("\n失败原因统计 (end-of-episode criteria, 可多重叠加):")
    for key, label in labels.items():
        if key in stats and stats[key] > 0:
            print(f"  {label}: {stats[key]}/{n}")


def _resolve_video_dir(trace_dir: Path | None, video_dir: str | None) -> Path:
    if video_dir is not None:
        return Path(video_dir)
    if trace_dir is not None:
        return trace_dir / "videos"
    return Path("./episode_videos_openpi")


def _confirm_output_dirs(
    *,
    trace_dir: Path | None,
    video_dir: Path,
) -> None:
    """Before a rollout run: if output dirs exist, ask whether to delete them."""
    candidates: list[tuple[str, Path]] = [("videos", video_dir)]
    if trace_dir is not None:
        candidates.insert(0, ("trace", trace_dir))

    existing = [(label, path) for label, path in candidates if path.exists()]
    if not existing:
        return

    print("Output directory(ies) already exist:")
    for label, path in existing:
        print(f"  {label}: {path.resolve()}")
    while True:
        choice = input("Delete and rewrite? [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            # Deepest paths first: video-dir may live under trace-dir.
            for _, path in sorted(
                existing, key=lambda item: len(item[1].resolve().parts), reverse=True
            ):
                if path.exists():
                    shutil.rmtree(path)
            print("Deleted; new outputs will be written fresh.")
            return
        if choice in ("n", "no"):
            print("Keeping existing files; new outputs will be appended.")
            return
        print("Please enter y or n.")


def _print_success_criteria(env, *, label: str = "end") -> None:
    """Print teleop save criteria breakdown (xy / z / gripper / ee_away)."""
    c = env.success_criteria()
    flags = (
        ("xy_ok", c["xy_ok"]),
        ("z_ok", c["z_ok"]),
        ("gripper_open", c["gripper_open"]),
        ("ee_away", c["ee_away"]),
    )
    failed = [name for name, ok in flags if not ok]
    print(f"  success criteria ({label}): " + " ".join(f"{n}={v}" for n, v in flags))
    if failed:
        print(f"    failed: {', '.join(failed)}")
    print(
        f"    xy_dist={c['xy_dist_m']:.4f}m tol={c['place_tol_xy_m']:.4f}m "
        f"cube_z={c['cube_z_m']:.4f}m z_min={c['z_min_m']:.4f}m"
    )
    print(
        f"    rh_r1={c['rh_r1']:.2e} (open<{c['gripper_open_thresh']:.1e}) "
        f"ee_above_cube={c['ee_z_above_cube_m']:.4f}m (need>{c['ee_away_thresh_m']:.2f}m)"
    )


def run_gt_replay(args: argparse.Namespace, trace_dir: Path | None) -> None:
    """Execute dataset GT actions in MuJoCo (diagnose sim execution vs policy)."""
    from core.my_env import MyEnv
    from core.videos.episode_video_recorder import EpisodeVideoRecorder

    episode_id = int(args.replay_gt_episode)
    states, actions, obj_init = _load_gt_episode(args.lerobot_root, episode_id)
    n_frames = len(actions)
    if args.replay_max_frames is not None:
        n_frames = min(n_frames, int(args.replay_max_frames))
        states = states[:n_frames]
        actions = actions[:n_frames]

    ee_cmd = policy_ee_pose_command(args.action_type)
    env = MyEnv(
        args.xml_path,
        seed=args.seed,
        action_type=args.action_type,
        state_type="qpos",
        ee_pose_command=ee_cmd,
    )
    _log_xml_arm_gains(env)
    print(
        f"GT replay ep {episode_id}: {n_frames} frames from {os.path.expanduser(args.lerobot_root)}"
    )
    print(
        f"action_type={env.action_type} ee_pose_command={env.ee_pose_command} "
        f"physics_settle={_physics_settle_desc(args)} "
        f"dataset_init={not args.no_replay_dataset_init and obj_init is not None}"
    )

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        with open(trace_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)

    video_recorder = EpisodeVideoRecorder(
        output_dir=args.video_dir,
        fps=HZ,
    )
    trace = EpisodeTrace(episode_id, trace_dir) if trace_dir is not None else None

    _dataset_aligned_reset(
        env,
        seed=args.seed,
        state0=states[0],
        obj_init=obj_init,
        use_recorded_layout=not args.no_replay_dataset_init,
        warmup_steps=int(args.dataset_init_warmup_steps),
    )

    video_recorder.start_episode(episode_id)
    step = 0
    episode_success = False
    cube_fell_logged = False
    track_errs: list[float] = []
    cmd_norms: list[float] = []

    # 0.tele.py body per 20Hz tick: step_env → pre_state → grab → act → step_env → render
    while env.env.is_viewer_alive() and step < n_frames:
        env.step_env()
        if not env.env.loop_every(HZ=HZ):
            continue

        state_pre = np.array(env.get_joint_state(), dtype=np.float32)
        ee_pre = _as_state(env.get_ee_pose())
        agent_raw, wrist_raw = env.grab_image()
        video_recorder.record_frame(agent_raw, wrist_raw)

        action_np = _apply_val_gripper_boost(
            np.asarray(actions[step], dtype=np.float64), args
        )
        if args.action_type == "ee_pose":
            assert ee_pre is not None
            cmd_norms.append(float(np.linalg.norm(_pose_delta_6d(action_np, ee_pre))))
        else:
            cmd_norms.append(float(np.linalg.norm(action_np[:6] - state_pre[:6])))

        env.step(action_np)
        _physics_settle(env, args)
        qpos_post = np.array(env.get_joint_state(), dtype=np.float32)
        ee_post = _as_state(env.get_ee_pose())
        if args.action_type == "ee_pose":
            assert ee_post is not None
            track_errs.append(float(np.linalg.norm(_pose_delta_6d(action_np, ee_post))))
        else:
            track_errs.append(float(np.linalg.norm(action_np[:6] - qpos_post[:6])))

        if trace is not None:
            trace.record_step(
                step_idx=step,
                qpos_pre=state_pre,
                qpos_post=qpos_post,
                action_executed=action_np,
                action_type=args.action_type,
                ee_pre=ee_pre,
                ee_post=ee_post,
                replan=False,
                infer_ms=0.0,
            )

        env.render(teleop=args.teleop_render)
        step += 1

        p_cube = cube_pose(env)
        on_table = cube_on_table(p_cube)
        if args.log_cube_every > 0 and step % args.log_cube_every == 0:
            print(
                f"[gt ep {episode_id} step {step}] cube xyz={p_cube.round(4)} "
                f"on_table={on_table} track_err={track_errs[-1]:.6f}",
                flush=True,
            )
        if not on_table and not cube_fell_logged:
            cube_fell_logged = True
            print(
                f"[gt ep {episode_id} step {step}] cube left table: xyz={p_cube.round(4)}",
                flush=True,
            )
        if env.check_success():
            print(f"GT replay ep {episode_id}: success in {step} steps")
            episode_success = True
            break

    video_recorder.stop(success=episode_success)

    cmd_mean = float(np.mean(cmd_norms)) if cmd_norms else 0.0
    track_mean = float(np.mean(track_errs)) if track_errs else 0.0
    track_max = float(np.max(track_errs)) if track_errs else 0.0
    print("-" * 30)
    print(f"GT replay done: steps={step} success={episode_success}")
    _print_success_criteria(env, label="after last frame")
    if args.action_type == "ee_pose":
        print(f"  |action-ee_pre| pose L2 mean={cmd_mean:.6f} (dataset label scale)")
        print(f"  |action-ee_post| pose L2 mean={track_mean:.6f} max={track_max:.6f}")
    else:
        print(f"  |action-state_pre| arm mean={cmd_mean:.6f} (dataset label scale)")
        print(f"  |action-qpos_post| arm mean={track_mean:.6f} max={track_max:.6f}")
    if track_mean > 0.01:
        print(
            "  -> Large tracking error: sim actuation likely not matching recorded targets."
        )
    elif track_mean < 0.001:
        print(
            "  -> Tracking OK: if arm still looked wrong, check init pose / obj_init / cameras."
        )
    print(f"Videos: {args.video_dir}")
    if trace_dir is not None:
        summary = trace.finalize(episode_success, step, action_type=args.action_type)
        summary["gt_replay"] = True
        summary["lerobot_root"] = args.lerobot_root
        with open(trace_dir / "gt_replay_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Traces: {trace_dir}")
        analyze_trace_dir(trace_dir)
        plot_paths = plot_trace_dir_qpos(trace_dir)
        if plot_paths:
            print(
                f"Qpos plots: {_resolve_qpos_plot_dir(trace_dir)} ({len(plot_paths)} png)"
            )
    print("-" * 30)


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    args.video_dir = str(_resolve_video_dir(trace_dir, args.video_dir))

    if args.trace_plot_only:
        if trace_dir is None:
            raise SystemExit("--trace-plot-only requires --trace-dir")
        paths = plot_trace_dir_qpos(trace_dir)
        if not paths:
            raise SystemExit(f"No rollout.npz under {trace_dir}")
        print(f"Plotted {len(paths)} episode(s) -> {_resolve_qpos_plot_dir(trace_dir)}")
        for p in paths:
            print(f"  {p}")
        return

    if args.trace_analyze_only:
        if trace_dir is None:
            raise SystemExit("--trace-analyze-only requires --trace-dir")
        analyze_trace_dir(trace_dir)
        plot_trace_dir_qpos(trace_dir)
        return

    _confirm_output_dirs(trace_dir=trace_dir, video_dir=Path(args.video_dir))

    if args.replay_gt_episode is not None:
        run_gt_replay(args, trace_dir)
        return

    # These imports are only needed for rollout execution (not trace-only analysis).
    from openpi_client import websocket_client_policy
    from core.my_env import MyEnv
    from core.videos.episode_video_recorder import EpisodeVideoRecorder

    if args.action_delta_stride > 1 and args.replan_steps > 1:
        raise ValueError(
            "--action-delta-stride>1 always uses chunk[0] only; set --replan-steps 1"
        )
    _validate_qpos_closed_loop_exec(args)
    _apply_eval_runtime_options(args)

    print(f"Connecting to openpi server at {args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port
    )
    metadata = client.get_server_metadata()
    print(f"Server metadata: {metadata}")

    ee_cmd = policy_ee_pose_command(args.action_type)
    env = MyEnv(
        args.xml_path,
        seed=args.seed,
        action_type=args.action_type,
        state_type="qpos",
        ee_pose_command=ee_cmd,
    )
    _log_xml_arm_gains(env)
    ee_guard = args.action_type == "ee_pose" and not args.no_ee_guard
    print(
        f"action_type: {env.action_type}, state_type: {env.state_type}, "
        f"ee_pose_command: {env.ee_pose_command}"
    )
    if ee_guard:
        print(
            f"EE guard ON: max_xyz_step={args.max_ee_xyz_step} max_rpy_step={args.max_ee_rpy_step}"
        )
    elif args.action_type == "ee_pose":
        print("EE guard OFF (--no-ee-guard)")

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        with open(trace_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)
        print(f"Tracing enabled -> {trace_dir}")

    video_recorder = EpisodeVideoRecorder(
        output_dir=args.video_dir,
        fps=HZ,
    )

    successful_episodes = 0
    all_summaries: list[dict] = []
    episode_outcomes: list[dict] = []
    print(
        f"Starting evaluation: {args.num_episodes} episodes, "
        f"replan_steps={args.replan_steps}, action_delta_stride={args.action_delta_stride}, "
        f"physics_settle={_physics_settle_desc(args)}, "
        f"policy_obs_source={args.policy_obs_source}"
    )
    if args.post_action_wait_s > 0.0:
        print(
            f"wait-replan mode: step(action) → wait {args.post_action_wait_s}s sim → "
            f"observe → infer (every step)"
        )
    if args.action_delta_stride > 1:
        print(
            f"k10 mode: infer every {args.action_delta_stride} sim steps, "
            f"hold chunk[0] (q_{{t+{args.action_delta_stride}}}) each cycle"
        )

    dataset_obs: _DatasetPolicyObsStream | None = None
    if args.policy_obs_source in _POLICY_OBS_USES_DATASET:
        obs_ep = int(args.policy_obs_dataset_episode)
        if args.dataset_init_episode is None:
            args.dataset_init_episode = obs_ep
            print(
                f"Auto --dataset-init-episode {obs_ep} "
                f"(match --policy-obs-dataset-episode for layout alignment)"
            )
        dataset_obs = _DatasetPolicyObsStream(args.lerobot_root, obs_ep)
        if args.policy_obs_source == "dataset":
            print(
                f"Policy observations from LeRobot ep {dataset_obs.episode_id} "
                f"({len(dataset_obs)} frames) — same tensors as training dataloader repack"
            )
        elif args.policy_obs_source == "state-dataset-image-sim":
            print(
                f"Policy obs hybrid A: state←LeRobot ep {dataset_obs.episode_id}, "
                f"image←sim ({len(dataset_obs)} frames)"
            )
        else:
            print(
                f"Policy obs hybrid B: state←sim, image←LeRobot ep {dataset_obs.episode_id} "
                f"({len(dataset_obs)} frames)"
            )

    if args.policy_obs_state_alpha < 0.0 or args.policy_obs_state_alpha > 1.0:
        raise ValueError(
            f"--policy-obs-state-alpha must be in [0, 1], got {args.policy_obs_state_alpha}"
        )
    if args.policy_obs_state_alpha > 0.0:
        if args.policy_obs_source != "sim":
            raise ValueError(
                "--policy-obs-state-alpha only applies to --policy-obs-source sim"
            )
        if args.dataset_init_episode is None:
            raise ValueError(
                "--policy-obs-state-alpha requires --dataset-init-episode for demo qpos"
            )

    ref_states: np.ndarray | None = None
    if args.dataset_init_episode is not None:
        ref_states, _, _ = _load_gt_episode(
            args.lerobot_root, int(args.dataset_init_episode)
        )
        if args.policy_obs_source == "sim":
            print(
                f"Sim closed-loop ref: demo ep {args.dataset_init_episode} "
                f"({len(ref_states)} frames) — log sim-vs-demo drift every 10 steps"
            )
            if args.max_steps > len(ref_states):
                print(
                    f"Auto --max-steps {len(ref_states)} "
                    f"(match demo ep length; was {args.max_steps})"
                )
                args.max_steps = len(ref_states)

    if args.policy_obs_state_alpha > 0.0 and ref_states is not None:
        print(
            f"Policy state blend: (1-{args.policy_obs_state_alpha:.2f})*sim + "
            f"{args.policy_obs_state_alpha:.2f}*demo ep {args.dataset_init_episode}"
        )

    for episode in range(args.num_episodes):
        if args.dataset_init_episode is not None:
            init_ep = int(args.dataset_init_episode)
            states0, _, obj_init = _load_gt_episode(args.lerobot_root, init_ep)
            _dataset_aligned_reset(
                env,
                seed=args.seed,
                state0=states0[0],
                obj_init=obj_init,
                use_recorded_layout=True,
                warmup_steps=int(args.dataset_init_warmup_steps),
            )
        else:
            env.reset()
        video_recorder.start_episode(episode)

        do_trace = trace_dir is not None and (
            args.trace_episodes == 0 or episode < args.trace_episodes
        )
        trace = (
            EpisodeTrace(episode, trace_dir, ref_states=ref_states)
            if do_trace
            else None
        )

        if args.post_action_wait_s > 0.0:
            episode_success, step, cube_fell_logged = _run_wait_replan_episode(
                env,
                client,
                args,
                episode,
                dataset_obs=dataset_obs,
                ref_states=ref_states,
                trace=trace,
                video_recorder=video_recorder,
                ee_guard=ee_guard,
            )
            if episode_success:
                successful_episodes += 1
        else:
            action_plan: collections.deque[np.ndarray] = collections.deque()
            active_chunk: np.ndarray | None = None
            chunk_cursor = 0
            replan_state_pre: np.ndarray | None = None
            qpos_hold_anchor: np.ndarray | None = None
            qpos_hold_target: np.ndarray | None = None
            step = 0
            episode_success = False
            cube_fell_logged = False

            # 0.tele.py: step_env → loop_every → obs → (infer) → step → step_env → render
            while env.env.is_viewer_alive() and step < args.max_steps:
                env.step_env()
                if not env.env.loop_every(HZ=HZ):
                    continue

                sim_qpos_pre = np.array(env.get_joint_state(), dtype=np.float32)
                if (
                    args.policy_obs_source == "sim"
                    and ref_states is not None
                    and step < len(ref_states)
                    and step % 10 == 0
                ):
                    drift = float(
                        np.linalg.norm(sim_qpos_pre[:6] - ref_states[step, :6])
                    )
                    print(
                        f"[ep {episode + 1} step {step}] sim-vs-demo drift={drift:.4f} rad",
                        flush=True,
                    )

                if args.policy_obs_source == "dataset":
                    if step >= len(dataset_obs):
                        print(
                            f"Episode {episode + 1}: dataset obs exhausted at step {step} "
                            f"(len={len(dataset_obs)})"
                        )
                        break
                    element = dataset_obs.obs_at(step, args.prompt)
                    state_pre = _as_state(element["observation/state"])
                    agent_img = element["observation/image"]
                    wrist_img = element["observation/wrist_image"]
                    # Policy obs from LeRobot; live sim cameras for viewer overlay + video.
                    agent_raw, wrist_raw = env.grab_image()
                elif args.policy_obs_source == "state-dataset-image-sim":
                    if step >= len(dataset_obs):
                        print(
                            f"Episode {episode + 1}: dataset obs exhausted at step {step} "
                            f"(len={len(dataset_obs)})"
                        )
                        break
                    _, agent_img, wrist_img, agent_raw, wrist_raw, _ = (
                        _tele_observation_frame(env, args.prompt)
                    )
                    ds_element = dataset_obs.obs_at(step, args.prompt)
                    state_pre = _as_state(ds_element["observation/state"])
                    element = {
                        "observation/image": agent_img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state_pre,
                        "prompt": args.prompt,
                    }
                elif args.policy_obs_source == "state-sim-image-dataset":
                    if step >= len(dataset_obs):
                        print(
                            f"Episode {episode + 1}: dataset obs exhausted at step {step} "
                            f"(len={len(dataset_obs)})"
                        )
                        break
                    state_pre, _, _, agent_raw, wrist_raw, sim_element = (
                        _tele_observation_frame(env, args.prompt)
                    )
                    ds_element = dataset_obs.obs_at(step, args.prompt)
                    agent_img = ds_element["observation/image"]
                    wrist_img = ds_element["observation/wrist_image"]
                    element = {
                        "observation/image": agent_img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": sim_element["observation/state"],
                        "prompt": args.prompt,
                    }
                else:
                    state_pre, agent_img, wrist_img, agent_raw, wrist_raw, element = (
                        _tele_observation_frame(env, args.prompt)
                    )
                    if (
                        args.policy_obs_state_alpha > 0.0
                        and ref_states is not None
                        and step < len(ref_states)
                    ):
                        state_pre = _blend_policy_state(
                            state_pre,
                            ref_states[step],
                            args.policy_obs_state_alpha,
                        )
                        element["observation/state"] = state_pre

                video_recorder.record_frame(agent_raw, wrist_raw)
                ee_pre = _as_state(env.get_ee_pose())

                infer_ms = 0.0
                replan = False
                need_replan = (
                    not action_plan
                    if args.action_delta_stride > 1
                    else (
                        active_chunk is None
                        or chunk_cursor >= _qpos_chunk_horizon(args, len(active_chunk))
                    )
                )
                if need_replan:
                    replan = True
                    print(
                        f"[ep {episode + 1} step {step}] Calling policy infer "
                        f"(first call may take 1-10 min for JAX compile)...",
                        flush=True,
                    )
                    infer_out = client.infer(element)
                    action_chunk = infer_out["actions"]
                    infer_ms = float(
                        infer_out.get("policy_timing", {}).get("infer_ms", 0.0)
                    )
                    chunk = np.asarray(action_chunk, dtype=np.float32)
                    if chunk.ndim == 1:
                        chunk = chunk[np.newaxis, :]
                    if args.action_type == "ee_pose" and ee_pre is not None:
                        delta0 = _pose_delta_6d(chunk[0], ee_pre)
                        delta_msg = f"ee_delta[0]={delta0} |xyz|={np.linalg.norm(delta0[:3]):.4f}"
                    elif (
                        args.qpos_chunk_integrate
                        and args.qpos_receding_horizon
                        and args.action_type == "qpos"
                    ):
                        delta0 = chunk[0, :6] - state_pre[:6]
                        delta_msg = f"qpos_integrate Δ[0]={delta0} |Δ|={np.linalg.norm(delta0):.4f}"
                    else:
                        delta_msg = f"qpos_delta[0]={(chunk[0, :6] - state_pre[:6])}"
                    print(
                        f"[ep {episode + 1} step {step}] infer done: "
                        f"chunk={chunk.shape} infer_ms={infer_ms:.0f} {delta_msg}",
                        flush=True,
                    )
                    if chunk.shape[-1] < 7:
                        raise ValueError(
                            f"Expected actions with last dim >= 7, got shape {chunk.shape}"
                        )
                    if trace is not None:
                        trace.record_replan(
                            step,
                            state_pre,
                            chunk,
                            infer_ms,
                            action_type=args.action_type,
                            ee_pre=ee_pre,
                        )
                    if args.action_delta_stride > 1:
                        # Plan C: chunk[0] = target at t+K; hold for K ticks (do not use chunk[1..]).
                        target = np.asarray(chunk[0, :7], dtype=np.float64)
                        if args.action_type == "qpos":
                            qpos_hold_anchor = np.asarray(
                                state_pre[:6], dtype=np.float64
                            )
                            qpos_hold_target = target[:6].copy()
                        for _ in range(args.action_delta_stride):
                            action_plan.append(target.copy())
                    else:
                        active_chunk = chunk
                        chunk_cursor = 0
                        replan_state_pre = np.asarray(state_pre, dtype=np.float32)

                if args.action_delta_stride > 1:
                    action_np = action_plan.popleft()
                else:
                    chunk_idx = chunk_cursor
                    action_np = np.asarray(
                        active_chunk[chunk_idx, :7], dtype=np.float64
                    )
                    if (
                        args.qpos_chunk_integrate
                        and args.qpos_receding_horizon
                        and args.action_type == "qpos"
                    ):
                        if replan_state_pre is None:
                            raise RuntimeError(
                                "replan_state_pre missing for qpos chunk integrate"
                            )
                        action_np = _qpos_chunk_integrated_cmd(
                            active_chunk,
                            chunk_idx,
                            replan_state_pre,
                            sim_qpos_pre,
                        )
                    chunk_cursor += 1
                if (
                    args.action_type == "qpos"
                    and args.action_delta_stride > 1
                    and args.qpos_hold_ramp
                    and qpos_hold_anchor is not None
                    and qpos_hold_target is not None
                ):
                    hold_remaining = len(action_plan) + 1
                    hold_step = args.action_delta_stride - hold_remaining
                    action_np = np.asarray(action_np, dtype=np.float64).copy()
                    action_np[:6] = _qpos_hold_setpoint(
                        qpos_hold_anchor,
                        qpos_hold_target,
                        args.action_delta_stride,
                        hold_step,
                    )
                if ee_guard:
                    if ee_pre is None:
                        ee_pre = _as_state(env.get_ee_pose())
                    action_np, clipped = clamp_absolute_ee_action(
                        action_np,
                        ee_pre,
                        max_xyz_step=args.max_ee_xyz_step,
                        max_rpy_step=args.max_ee_rpy_step,
                    )
                    if clipped and step % 10 == 0:
                        print(
                            f"[ep {episode + 1} step {step}] EE guard clipped policy target"
                        )

                action_np = _apply_val_gripper_boost(
                    np.asarray(action_np, dtype=np.float64), args
                )
                env.step(action_np)
                _physics_settle(env, args)
                qpos_post = _as_state(env.get_joint_state())
                ee_post = _as_state(env.get_ee_pose())

                if trace is not None:
                    trace.record_step(
                        step_idx=step,
                        qpos_pre=state_pre,
                        qpos_post=qpos_post,
                        action_executed=action_np,
                        action_type=args.action_type,
                        ee_pre=ee_pre,
                        ee_post=ee_post,
                        replan=replan,
                        infer_ms=infer_ms,
                        sim_qpos_pre=sim_qpos_pre,
                    )

                env.render(teleop=args.teleop_render)
                step += 1

                p_cube = cube_pose(env)
                on_table = cube_on_table(p_cube)
                if args.log_cube_every > 0 and step % args.log_cube_every == 0:
                    print(
                        f"[ep {episode + 1} step {step}] cube xyz={p_cube.round(4)} on_table={on_table}",
                        flush=True,
                    )
                if not on_table and not cube_fell_logged:
                    cube_fell_logged = True
                    print(
                        f"[ep {episode + 1} step {step}] cube left table: xyz={p_cube.round(4)}",
                        flush=True,
                    )

            if env.check_success():
                print(f"Episode {episode + 1}: success in {step} steps")
                episode_success = True
                successful_episodes += 1
                break

        video_recorder.stop(success=episode_success)

        outcome = _episode_outcome(
            env,
            success=episode_success,
            num_steps=step,
            max_steps=args.max_steps,
            cube_left_table=cube_fell_logged,
        )

        if trace is not None:
            summary = trace.finalize(
                episode_success, step, action_type=args.action_type
            )
            summary.update(outcome)
            with open(trace.ep_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            all_summaries.append(summary)
            msg = (
                f"  trace ep {episode + 1}: cmd_norm={summary['arm_cmd_norm_mean']:.6f} "
                f"near_zero={summary['arm_cmd_near_zero_frac']:.1%} "
                f"track_err={summary['arm_track_err_mean']:.6f}"
            )
            if "ee_xyz_cmd_norm_mean" in summary:
                msg += (
                    f" xyz_cmd={summary['ee_xyz_cmd_norm_mean']:.6f} "
                    f"ee_motion={summary['ee_motion_norm_mean']:.6f}"
                )
            if "sim_state_drift_mean" in summary:
                msg += f" sim_drift={summary['sim_state_drift_mean']:.4f}"
            if not episode_success:
                msg += (
                    f" outcome={outcome['outcome']} failed={outcome['failed_criteria']}"
                )
            print(msg)
        else:
            summary = {
                "episode_index": episode,
                "success": episode_success,
                "num_steps": step,
                **outcome,
            }

        episode_outcomes.append(
            {
                "episode_index": episode,
                "success": episode_success,
                "num_steps": step,
                **outcome,
            }
        )

        if not episode_success and env.env.is_viewer_alive():
            print(f"Episode {episode + 1}: failure — {outcome['outcome']}")
            _print_success_criteria(env, label=f"ep {episode + 1} end")

        if not env.env.is_viewer_alive():
            print("Viewer closed; stopping evaluation.")
            break

    total_evaluated = min(episode + 1, args.num_episodes)
    success_rate = (
        (successful_episodes / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    )

    if trace_dir is not None and all_summaries:
        with open(trace_dir / "all_episodes_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)

    eval_summary = {
        "num_episodes": total_evaluated,
        "successes": successful_episodes,
        "success_rate_pct": success_rate,
        "failure_stats": _aggregate_failure_stats(episode_outcomes),
        "episodes": episode_outcomes,
        "action_type": args.action_type,
        "policy_obs_source": args.policy_obs_source,
        "policy_obs_state_alpha": args.policy_obs_state_alpha,
        "action_delta_stride": args.action_delta_stride,
        "replan_steps": args.replan_steps,
        "qpos_chunk_integrate": args.qpos_chunk_integrate,
        "qpos_receding_horizon": args.qpos_receding_horizon,
        "teleop_tick": args.teleop_tick,
        "post_action_wait_s": args.post_action_wait_s,
        "physics_settle": _physics_settle_desc(args),
        "dataset_init_episode": args.dataset_init_episode,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    summary_path = (
        trace_dir / "eval_summary.json"
        if trace_dir is not None
        else Path(args.video_dir) / "eval_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("-" * 30)
    print("Evaluation done")
    print(f"Episodes: {total_evaluated}")
    print(f"Successes: {successful_episodes}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Summary JSON: {summary_path}")
    _print_failure_breakdown(episode_outcomes)
    print(f"Videos: {args.video_dir}")
    if trace_dir is not None:
        print(f"Traces: {trace_dir}")
        analyze_trace_dir(trace_dir)
        plot_paths = plot_trace_dir_qpos(trace_dir)
        if plot_paths:
            print(
                f"Qpos plots: {_resolve_qpos_plot_dir(trace_dir)} ({len(plot_paths)} png)"
            )
    print("-" * 30)


if __name__ == "__main__":
    main()
