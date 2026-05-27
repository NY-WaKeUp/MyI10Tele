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
     --trace-dir ./openpi_eval_trace

4. Inspect traces:

   PYTHONPATH=src python src/core/8.val_openpi_sim.py --trace-dir ./openpi_eval_trace --trace-analyze-only
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

from core.dataset_config import policy_ee_pose_command

# MuJoCo viewer display (change if needed)
os.environ.setdefault("DISPLAY", ":51.0")

TASK_NAME = "Put cube on the black platform"
XML_PATH = "/home/ningyu/MyI10Tele/assets/aubo_i10_inspire/myscene.xml"
HZ = 20


def _preprocess_image(img: np.ndarray, size: int) -> np.ndarray:
    # Import lazily so `--trace-analyze-only` works without openpi-client installed.
    from openpi_client import image_tools

    img = np.asarray(img)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


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
    d = np.asarray(targets, dtype=np.float32)[:, :6] - np.asarray(refs, dtype=np.float32)[:, :6]
    d[:, 3:6] = _wrap_pi(d[:, 3:6]).astype(np.float32)
    return d


def _build_observation(
    agent_img: np.ndarray,
    wrist_img: np.ndarray,
    state: np.ndarray,
    prompt: str,
    resize_size: int,
) -> dict:
    return {
        "observation/image": _preprocess_image(agent_img, resize_size),
        "observation/wrist_image": _preprocess_image(wrist_img, resize_size),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }


class EpisodeTrace:
    """Per-episode rollout log for debugging policy I/O vs simulator state."""

    def __init__(self, episode_index: int, trace_dir: Path, save_images: bool) -> None:
        self.episode_index = episode_index
        self.ep_dir = trace_dir / f"episode_{episode_index:03d}"
        self.ep_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self._img_count = 0

        self.step: list[int] = []
        # qpos (7D) always available from env.get_joint_state()
        self.qpos_pre: list[np.ndarray] = []
        self.qpos_post: list[np.ndarray] = []
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

    def maybe_save_images(self, agent_img: np.ndarray, wrist_img: np.ndarray, step_idx: int) -> None:
        if not self.save_images or self._img_count >= 30:
            return
        import cv2

        cv2.imwrite(str(self.ep_dir / f"step_{step_idx:04d}_agent.png"), agent_img)
        cv2.imwrite(str(self.ep_dir / f"step_{step_idx:04d}_wrist.png"), wrist_img)
        self._img_count += 1

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
            arm_delta_chunk = (chunk6 - np.asarray(qpos_pre, dtype=np.float32)[:6]).astype(np.float32)
        elif action_type == "ee_pose":
            if ee_pre is None:
                raise ValueError("ee_pre required when action_type=='ee_pose'")
            ref = np.asarray(ee_pre, dtype=np.float32)[:6]
            arm_delta_chunk = np.stack([_pose_delta_6d(row, ref) for row in chunk6], axis=0)
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
                arm_track_err = np.linalg.norm(actions[:, :6] - qpos_post[:, :6], axis=1)
                arm_range_ref = qpos_pre
            elif action_type == "ee_pose":
                ee_pre_st = np.stack(self.ee_pre) if self.ee_pre else np.zeros((0, 7), dtype=np.float32)
                ee_post_st = np.stack(self.ee_post) if self.ee_post else np.zeros((0, 7), dtype=np.float32)
                arm_delta = _pose_delta_batch(actions, ee_pre_st)
                arm_track_err = np.linalg.norm(_pose_delta_batch(actions, ee_post_st), axis=1)
                arm_range_ref = ee_pre_st
                ee_motion_norm = np.linalg.norm(_pose_delta_batch(ee_post_st, ee_pre_st), axis=1)
                ee_xyz_cmd_norm = np.linalg.norm(arm_delta[:, :3], axis=1)
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
            qpos_pre = qpos_post = actions = arm_delta = arm_track_err = arm_cmd_norm = np.zeros((0,))
            arm_range_ref = np.zeros((0,))
            ee_motion_norm = ee_xyz_cmd_norm = None

        np.savez(
            self.ep_dir / "rollout.npz",
            step=np.asarray(self.step, dtype=np.int32),
            qpos_pre=qpos_pre,
            qpos_post=qpos_post,
            ee_pre=np.stack(self.ee_pre) if self.ee_pre else np.zeros((0, 7), dtype=np.float32),
            ee_post=np.stack(self.ee_post) if self.ee_post else np.zeros((0, 7), dtype=np.float32),
            action_executed=actions,
            arm_delta_cmd=arm_delta,
            arm_track_err=arm_track_err,
            arm_cmd_norm=arm_cmd_norm,
            gripper_cmd=np.asarray(self.gripper_cmd, dtype=np.float32),
            gripper_state_pre=np.asarray(self.gripper_state_pre, dtype=np.float32),
            gripper_state_post=np.asarray(self.gripper_state_post, dtype=np.float32),
            replan=np.asarray(self.replan, dtype=bool),
            infer_ms=np.asarray(self.infer_ms, dtype=np.float32),
            ee_motion_norm=ee_motion_norm if ee_motion_norm is not None else np.zeros((0,), dtype=np.float32),
            ee_xyz_cmd_norm=ee_xyz_cmd_norm if ee_xyz_cmd_norm is not None else np.zeros((0,), dtype=np.float32),
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
            "arm_cmd_norm_mean": float(arm_cmd_norm.mean()) if len(arm_cmd_norm) else 0.0,
            "arm_cmd_norm_max": float(arm_cmd_norm.max()) if len(arm_cmd_norm) else 0.0,
            "arm_cmd_near_zero_frac": float((arm_cmd_norm < cmd_near_zero_thresh).mean())
            if len(arm_cmd_norm)
            else 1.0,
            "arm_track_err_mean": float(arm_track_err.mean()) if len(arm_track_err) else 0.0,
            "arm_track_err_max": float(arm_track_err.max()) if len(arm_track_err) else 0.0,
            "gripper_toggles": int(np.sum(np.abs(np.diff(np.asarray(self.gripper_cmd))) > 0.5))
            if len(self.gripper_cmd) > 1
            else 0,
            "state_pre_range_arm": np.ptp(arm_range_ref[:, :6], axis=0).tolist() if len(arm_range_ref) else [],
            "action_range_arm": np.ptp(actions[:, :6], axis=0).tolist() if len(actions) else [],
        }
        if ee_xyz_cmd_norm is not None and len(ee_xyz_cmd_norm):
            summary["ee_xyz_cmd_norm_mean"] = float(ee_xyz_cmd_norm.mean())
            summary["ee_xyz_cmd_norm_max"] = float(ee_xyz_cmd_norm.max())
            summary["ee_xyz_near_zero_frac"] = float((ee_xyz_cmd_norm < 0.005).mean())
        if ee_motion_norm is not None and len(ee_motion_norm):
            summary["ee_motion_norm_mean"] = float(ee_motion_norm.mean())
        with open(self.ep_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


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
    print("\nPer-episode:")
    for r in rows:
        extra = ""
        if "ee_xyz_cmd_norm_mean" in r:
            extra = (
                f" xyz_cmd={r['ee_xyz_cmd_norm_mean']:.5f}"
                f" ee_motion={r.get('ee_motion_norm_mean', 0.0):.5f}"
            )
        print(
            f"  ep {r['episode_index']:03d}: steps={r['num_steps']} "
            f"cmd_norm={r['arm_cmd_norm_mean']:.5f} near_zero={r['arm_cmd_near_zero_frac']:.2%} "
            f"track_err={r['arm_track_err_mean']:.5f}{extra} "
            f"grip_toggles={r['gripper_toggles']} ok={r['success']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="openpi WebSocket eval on Aubo MyEnv sim")
    parser.add_argument("--host", type=str, default="localhost", help="openpi serve_policy host")
    parser.add_argument("--port", type=int, default=8000, help="openpi serve_policy port")
    parser.add_argument(
        "--resize-size",
        type=int,
        default=256,
        help="Image edge length sent to the policy (training data uses 256)",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=5,
        help="Execute this many actions from each chunk before re-inferring (action_horizon=10)",
    )
    parser.add_argument("--prompt", type=str, default=TASK_NAME, help="Language instruction for the policy")
    parser.add_argument("--xml-path", type=str, default=XML_PATH, help="MuJoCo scene XML")
    parser.add_argument("--seed", type=int, default=42, help="MyEnv RNG seed")
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of evaluation rollouts")
    parser.add_argument("--max-steps", type=int, default=600, help="Max control steps per episode")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./episode_videos_openpi",
        help="Directory for side-by-side rollout videos",
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
        default=3,
        help="Number of episodes to trace (from the start); 0 = trace all",
    )
    parser.add_argument(
        "--trace-save-images",
        action="store_true",
        help="Save first camera frames per traced episode (up to 30 per ep)",
    )
    parser.add_argument(
        "--trace-analyze-only",
        action="store_true",
        help="Print trace summary from --trace-dir and exit",
    )
    parser.add_argument(
        "--teleop-render",
        action="store_true",
        help="Match 0.tele.py viewer overlays (camera panels + key hints)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir) if args.trace_dir else None

    if args.trace_analyze_only:
        if trace_dir is None:
            raise SystemExit("--trace-analyze-only requires --trace-dir")
        analyze_trace_dir(trace_dir)
        return

    # These imports are only needed for rollout execution (not trace-only analysis).
    from openpi_client import websocket_client_policy
    from core.my_env import MyEnv
    from core.videos.episode_video_recorder import EpisodeVideoRecorder

    print(f"Connecting to openpi server at {args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
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
    print(
        f"action_type: {env.action_type}, state_type: {env.state_type}, "
        f"ee_pose_command: {env.ee_pose_command}"
    )

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        with open(trace_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)
        print(f"Tracing enabled -> {trace_dir}")

    video_recorder = EpisodeVideoRecorder(
        output_dir=args.video_dir,
        fps=HZ,
        frame_size=(512, 256),
    )

    successful_episodes = 0
    all_summaries: list[dict] = []
    print(f"Starting evaluation: {args.num_episodes} episodes, replan_steps={args.replan_steps}")

    for episode in range(args.num_episodes):
        env.reset()
        action_plan: collections.deque[np.ndarray] = collections.deque()
        step = 0
        episode_success = False
        video_recorder.start_episode(episode)

        do_trace = trace_dir is not None and (
            args.trace_episodes == 0 or episode < args.trace_episodes
        )
        trace = (
            EpisodeTrace(episode, trace_dir, save_images=args.trace_save_images)
            if do_trace
            else None
        )

        # Match 0.tele.py: step_env first, then read state / act / step_env / render.
        while env.env.is_viewer_alive() and step < args.max_steps:
            env.step_env()
            if not env.env.loop_every(HZ=HZ):
                continue

            agent_img, wrist_img = env.grab_image()
            # Refresh viewer while policy infers (otherwise window stays black for minutes).
            env.render(teleop=args.teleop_render)
            video_recorder.record_frame(agent_img, wrist_img)
            state_pre = _as_state(env.get_joint_state())
            ee_pre = _as_state(env.get_ee_pose()) if args.action_type == "ee_pose" else None

            infer_ms = 0.0
            replan = False
            if not action_plan:
                replan = True
                element = _build_observation(
                    agent_img, wrist_img, state_pre, args.prompt, args.resize_size
                )
                if trace is not None:
                    trace.maybe_save_images(agent_img, wrist_img, step)
                print(
                    f"[ep {episode + 1} step {step}] Calling policy infer "
                    f"(first call may take 1-10 min for JAX compile)...",
                    flush=True,
                )
                infer_out = client.infer(element)
                action_chunk = infer_out["actions"]
                infer_ms = float(infer_out.get("policy_timing", {}).get("infer_ms", 0.0))
                chunk = np.asarray(action_chunk, dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk[np.newaxis, :]
                if args.action_type == "ee_pose" and ee_pre is not None:
                    delta0 = _pose_delta_6d(chunk[0], ee_pre)
                    delta_msg = f"ee_delta[0]={delta0} |xyz|={np.linalg.norm(delta0[:3]):.4f}"
                else:
                    delta_msg = f"qpos_delta[0]={(chunk[0, :6] - state_pre[:6])}"
                print(
                    f"[ep {episode + 1} step {step}] infer done: "
                    f"chunk={chunk.shape} infer_ms={infer_ms:.0f} {delta_msg}",
                    flush=True,
                )
                if chunk.shape[-1] < 7:
                    raise ValueError(f"Expected actions with last dim >= 7, got shape {chunk.shape}")
                if trace is not None:
                    trace.record_replan(
                        step,
                        state_pre,
                        chunk,
                        infer_ms,
                        action_type=args.action_type,
                        ee_pre=ee_pre,
                    )
                n_take = min(args.replan_steps, len(chunk))
                for i in range(n_take):
                    action_plan.append(np.asarray(chunk[i, :7], dtype=np.float64))

            action_np = action_plan.popleft()
            env.step(action_np)
            # Advance sim so state_post matches training "actions = q after step".
            env.step_env()
            qpos_post = _as_state(env.get_joint_state())
            ee_post = _as_state(env.get_ee_pose()) if args.action_type == "ee_pose" else None

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
                )

            env.render(teleop=args.teleop_render)
            step += 1

            if env.check_success():
                print(f"Episode {episode + 1}: success in {step} steps")
                episode_success = True
                successful_episodes += 1
                break

        video_recorder.stop(success=episode_success)

        if trace is not None:
            summary = trace.finalize(episode_success, step, action_type=args.action_type)
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
            print(msg)

        if not episode_success and env.env.is_viewer_alive():
            print(f"Episode {episode + 1}: failure (max steps {args.max_steps})")

        if not env.env.is_viewer_alive():
            print("Viewer closed; stopping evaluation.")
            break

    total_evaluated = min(episode + 1, args.num_episodes)
    success_rate = (successful_episodes / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    if trace_dir is not None and all_summaries:
        with open(trace_dir / "all_episodes_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)

    print("-" * 30)
    print("Evaluation done")
    print(f"Episodes: {total_evaluated}")
    print(f"Successes: {successful_episodes}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Videos: {args.video_dir}")
    if trace_dir is not None:
        print(f"Traces: {trace_dir}")
        analyze_trace_dir(trace_dir)
    print("-" * 30)


if __name__ == "__main__":
    main()
