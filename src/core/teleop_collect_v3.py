import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mujoco import viewer
from pynput import keyboard

from src.core.lerobot_v3_writer import EpisodeBuffer, LeRobotV3Writer
from src.core.mujoco_teleop_env import MujocoTeleopEnv, default_model_path
from src.dataset.task_spec import TaskSpec


@dataclass
class KeyState:
    pressed: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard teleop collection with LeRobot v3 layout.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="local/aubo-i10-inspire")
    parser.add_argument("--task", type=str, default="Pick and place the colored cube.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=1500)
    parser.add_argument("--physics-fps", type=int, default=100)
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--pos-step", type=float, default=0.002)
    parser.add_argument("--rot-step-deg", type=float, default=1.0)
    parser.add_argument("--ik-damping", type=float, default=1e-3)
    parser.add_argument("--ik-gain", type=float, default=0.6)
    return parser.parse_args()


def _build_action(keys: set[str], pos_step: float, rot_step_rad: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[0] = pos_step * (int("d" in keys) - int("a" in keys))
    action[1] = pos_step * (int("w" in keys) - int("s" in keys))
    action[2] = pos_step * (int("r" in keys) - int("f" in keys))
    action[3] = rot_step_rad * (int("k" in keys) - int("i" in keys))
    action[4] = rot_step_rad * (int("l" in keys) - int("j" in keys))
    action[5] = rot_step_rad * (int("u" in keys) - int("o" in keys))
    action[6] = 0.01 * (int("[" in keys) - int("]" in keys))
    return action

def _print_help() -> None:
    print("=== Keyboard mapping ===")
    print("Move: w/s(+/-y), a/d(-/+x), r/f(+/-z)")
    print("Rotate: i/k(roll +/-), j/l(pitch +/-), u/o(yaw +/-)")
    print("Gripper: [ close, ] open")
    print("Episode: Enter save+next, Backspace discard+reset, Esc quit")


def main() -> None:
    args = parse_args()
    env = MujocoTeleopEnv(
        model_path=default_model_path(),
        seed=0,
        ik_damping=args.ik_damping,
        ik_gain=args.ik_gain,
    )

    data_hz = args.physics_fps
    sim_steps_per_control = max(1, int(round((1.0 / data_hz) / env.model.opt.timestep)))
    rot_step_rad = np.deg2rad(args.rot_step_deg)

    writer = LeRobotV3Writer(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        fps=data_hz,
        robot_type="aubo_i10_inspire_mujoco",
        action_dim=7,
        state_dim=7,
    )
    videos_root = Path(args.dataset_root) / "videos" / "main" / "chunk-000"

    key_state = KeyState(pressed=set())
    command = {"save_episode": False, "discard_episode": False, "quit": False}

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        if hasattr(key, "char") and key.char is not None:
            key_state.pressed.add(key.char.lower())
            return
        if key == keyboard.Key.enter:
            command["save_episode"] = True
        elif key == keyboard.Key.backspace:
            command["discard_episode"] = True
        elif key == keyboard.Key.esc:
            command["quit"] = True

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
        if hasattr(key, "char") and key.char is not None:
            key_state.pressed.discard(key.char.lower())

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    _print_help()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    task = TaskSpec(
        task_id="teleop_collect_v3",
        instruction=args.task,
        cube_xy_low=np.array([0.20, -1.15], dtype=np.float64),
        cube_xy_high=np.array([0.33, -1.03], dtype=np.float64),
        max_steps=args.steps_per_episode,
    )

    try:
        with viewer.launch_passive(env.model, env.data) as v:
            v.cam.azimuth = 0
            v.cam.elevation = -10
            v.cam.distance = 2.0
            v.cam.lookat[:] = np.array([0.4, -1.0, 0.43], dtype=np.float64)

            episode_saved = 0
            while v.is_running() and not command["quit"] and episode_saved < args.episodes:
                episode_idx = writer.new_episode_index()
                video_relpath = f"videos/main/chunk-000/file-{episode_idx:03d}.mp4"
                video_abspath = videos_root / f"file-{episode_idx:03d}.mp4"
                vw = cv2.VideoWriter(str(video_abspath), fourcc, args.video_fps, (args.width, args.height))

                states: list[np.ndarray] = []
                actions: list[np.ndarray] = []
                timestamps: list[float] = []
                t0 = time.perf_counter()

                env.reset(task)
                command["save_episode"] = False
                command["discard_episode"] = False

                for _ in range(args.steps_per_episode):
                    if command["quit"] or command["save_episode"] or command["discard_episode"]:
                        break

                    action = _build_action(key_state.pressed, args.pos_step, rot_step_rad)
                    out = env.step(action=action, sim_steps_per_control=sim_steps_per_control, task=task)
                    frame = env.render_main(width=args.width, height=args.height)
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    vw.write(bgr)
                    states.append(out.state)
                    actions.append(out.action)
                    timestamps.append(time.perf_counter() - t0)
                    v.sync()

                vw.release()

                if command["discard_episode"]:
                    video_abspath.unlink(missing_ok=True)
                    command["discard_episode"] = False
                    continue

                if len(states) == 0:
                    video_abspath.unlink(missing_ok=True)
                    continue

                writer.add_episode(
                    EpisodeBuffer(
                        episode_index=episode_idx,
                        task_index=0,
                        states=states,
                        actions=actions,
                        timestamps=timestamps,
                    ),
                    video_relpath=video_relpath,
                )
                episode_saved += 1
                command["save_episode"] = False
                print(f"[saved] episode={episode_idx}, steps={len(states)}, video={video_relpath}")

        writer.finalize(task_descriptions=[args.task])
        print(f"[done] dataset written to: {args.dataset_root}")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
