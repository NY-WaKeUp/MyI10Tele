#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate an openpi JAX checkpoint on Aubo MyEnv simulation via WebSocket.

Prerequisites
-------------
1. openpi policy server (separate terminal, openpi repo + .venv):

   cd /home/ningyu/code_before_paper/openpi
   # Use a free GPU only (e.g. GPU 0). Kill stale serve_policy if nvidia-smi shows ~18G used.
   CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
   uv run scripts/serve_policy.py \\
     --default-prompt "Put cube on the black platform" \\
     policy:checkpoint \\
     --policy.config=pi0_auboI10_low_mem_finetune \\
     --policy.dir=checkpoints/pi0_auboI10_low_mem_finetune/aubo_lora_v1/19999

2. openpi-client in this project's environment:

   uv pip install -e /home/ningyu/code_before_paper/openpi/packages/openpi-client

3. Run this script from MyI10Tele (MuJoCo / DISPLAY must work):

   cd /home/ningyu/MyI10Tele
   PYTHONPATH=src python src/core/8.val_openpi_sim.py
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from core.my_env import MyEnv
from core.videos.episode_video_recorder import EpisodeVideoRecorder

# MuJoCo viewer display (change if needed)
os.environ.setdefault("DISPLAY", ":17.0")

TASK_NAME = "Put cube on the black platform"
XML_PATH = "/home/ningyu/MyI10Tele/assets/aubo_i10_inspire/myscene.xml"
HZ = 20


def _preprocess_image(img: np.ndarray, size: int) -> np.ndarray:
    img = np.asarray(img)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


def _as_state(state) -> np.ndarray:
    return np.asarray(state, dtype=np.float32)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="openpi WebSocket eval on Aubo MyEnv sim")
    parser.add_argument("--host", type=str, default="localhost", help="openpi serve_policy host")
    parser.add_argument("--port", type=int, default=8000, help="openpi serve_policy port")
    parser.add_argument("--resize-size", type=int, default=224, help="Image edge length sent to the policy")
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
        "--legacy-pre-step-state",
        action="store_true",
        help="Use pre-step state/images (old eval bug); default matches 0.tele.py logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Connecting to openpi server at {args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    print(f"Server metadata: {metadata}")

    env = MyEnv(args.xml_path, seed=args.seed, action_type="qpos", state_type="qpos")
    print(f"action_type: {env.action_type}, state_type: {env.state_type}")

    video_recorder = EpisodeVideoRecorder(
        output_dir=args.video_dir,
        fps=HZ,
        frame_size=(512, 256),
    )

    successful_episodes = 0
    print(f"Starting evaluation: {args.num_episodes} episodes, replan_steps={args.replan_steps}")

    for episode in range(args.num_episodes):
        env.reset()
        action_plan: collections.deque[np.ndarray] = collections.deque()
        step = 0
        episode_success = False
        video_recorder.start_episode(episode)

        # Match 0.tele.py: observation.state is post-step; images are grabbed pre-step.
        state = _as_state(env.get_joint_state())
        prev_agent_img: np.ndarray | None = None
        prev_wrist_img: np.ndarray | None = None

        while env.env.is_viewer_alive() and step < args.max_steps:
            env.step_env()

            if not env.env.loop_every(HZ=HZ):
                continue

            agent_img, wrist_img = env.grab_image()
            video_recorder.record_frame(agent_img, wrist_img)

            if not action_plan:
                if args.legacy_pre_step_state:
                    infer_agent, infer_wrist = agent_img, wrist_img
                    infer_state = _as_state(env.get_joint_state())
                else:
                    infer_agent = (
                        prev_agent_img if prev_agent_img is not None else agent_img
                    )
                    infer_wrist = (
                        prev_wrist_img if prev_wrist_img is not None else wrist_img
                    )
                    infer_state = state

                element = _build_observation(
                    infer_agent, infer_wrist, infer_state, args.prompt, args.resize_size
                )
                action_chunk = client.infer(element)["actions"]
                chunk = np.asarray(action_chunk)
                if chunk.ndim == 1:
                    chunk = chunk[np.newaxis, :]
                if chunk.shape[-1] < 7:
                    raise ValueError(f"Expected actions with last dim >= 7, got shape {chunk.shape}")
                n_take = min(args.replan_steps, len(chunk))
                for i in range(n_take):
                    action_plan.append(np.asarray(chunk[i, :7], dtype=np.float64))

            action_np = action_plan.popleft()
            state = _as_state(env.step(action_np))
            prev_agent_img = agent_img
            prev_wrist_img = wrist_img
            env.render()
            step += 1

            if env.check_success():
                print(f"Episode {episode + 1}: success in {step} steps")
                episode_success = True
                successful_episodes += 1
                break

        video_recorder.stop(success=episode_success)

        if not episode_success and env.env.is_viewer_alive():
            print(f"Episode {episode + 1}: failure (max steps {args.max_steps})")

        if not env.env.is_viewer_alive():
            print("Viewer closed; stopping evaluation.")
            break

    total_evaluated = min(episode + 1, args.num_episodes)
    success_rate = (successful_episodes / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    print("-" * 30)
    print("Evaluation done")
    print(f"Episodes: {total_evaluated}")
    print(f"Successes: {successful_episodes}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Videos: {args.video_dir}")
    print("-" * 30)


if __name__ == "__main__":
    main()
