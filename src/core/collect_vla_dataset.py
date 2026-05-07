import argparse
import time
from pathlib import Path

import cv2
import glfw
import numpy as np
from mujoco import viewer

from src.core.mujoco_teleop_env import MujocoTeleopEnv, default_model_path
from src.dataset.lerobot_v3_dataset import EpisodeRecord, LeRobotV3DatasetWriter
from src.dataset.task_spec import TASK_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect teleop data for reach/grasp/place tasks.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="local/aubo-i10-inspire-vla")
    parser.add_argument("--episodes_per_task", type=int, default=30)
    parser.add_argument("--physics_fps", type=int, default=100)
    parser.add_argument("--video_fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos_step", type=float, default=0.002)
    parser.add_argument("--rot_step", type=float, default=0.017)
    parser.add_argument("--gripper_step", type=float, default=0.01)
    parser.add_argument("--ik_damping", type=float, default=1e-3)
    parser.add_argument("--ik_gain", type=float, default=0.6)
    return parser.parse_args()


def build_action(keys: set[str], pos_step: float, rot_step: float, grip_step: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[0] = pos_step * (int("d" in keys) - int("a" in keys))
    action[1] = pos_step * (int("w" in keys) - int("s" in keys))
    action[2] = pos_step * (int("r" in keys) - int("f" in keys))
    action[3] = rot_step * (int("k" in keys) - int("i" in keys))
    action[4] = rot_step * (int("l" in keys) - int("j" in keys))
    action[5] = rot_step * (int("u" in keys) - int("o" in keys))
    action[6] = grip_step * (int("[" in keys) - int("]" in keys))
    return action


def main() -> None:
    args = parse_args()
    env = MujocoTeleopEnv(default_model_path(), seed=args.seed, ik_damping=args.ik_damping, ik_gain=args.ik_gain)
    writer = LeRobotV3DatasetWriter(root=args.data_dir, fps=args.physics_fps, repo_id=args.repo_id, robot_type="aubo_i10_inspire")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    sim_steps_per_control = max(1, int(round((1.0 / args.physics_fps) / env.model.opt.timestep)))
    keys: set[str] = set()
    cmd = {"save": False, "discard": False, "quit": False}

    key_map = {
        glfw.KEY_W: "w",
        glfw.KEY_A: "a",
        glfw.KEY_S: "s",
        glfw.KEY_D: "d",
        glfw.KEY_R: "r",
        glfw.KEY_F: "f",
        glfw.KEY_I: "i",
        glfw.KEY_J: "j",
        glfw.KEY_K: "k",
        glfw.KEY_L: "l",
        glfw.KEY_U: "u",
        glfw.KEY_O: "o",
        glfw.KEY_LEFT_BRACKET: "[",
        glfw.KEY_RIGHT_BRACKET: "]",
    }

    def on_key(keycode: int) -> None:
        if keycode in key_map:
            key_name = key_map[keycode]
            if key_name in keys:
                keys.discard(key_name)
            else:
                keys.add(key_name)
            return
        if keycode in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            cmd["save"] = True
        elif keycode == glfw.KEY_BACKSPACE:
            cmd["discard"] = True
        elif keycode == glfw.KEY_ESCAPE:
            cmd["quit"] = True

    print("Controls (MuJoCo window): wasd/rf move, ijkluo rotate, [/ ] gripper, Enter save, Backspace discard, Esc quit")
    print("Movement keys are toggled by key events in the viewer window.")
    episode_index = 0
    target_total = args.episodes_per_task * len(TASK_SPECS)
    saved = 0

    with viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
            v.cam.azimuth = 0
            v.cam.elevation = -10
            v.cam.distance = 2.0
            v.cam.lookat[:] = np.array([0.4, -1.0, 0.43], dtype=np.float64)

            while v.is_running() and (not cmd["quit"]) and saved < target_total:
                task = TASK_SPECS[saved % len(TASK_SPECS)]
                print(f"[episode {episode_index}] task={task.task_id}: {task.instruction}")
                env.reset(task)
                cmd["save"] = False
                cmd["discard"] = False

                main_rel, wrist_rel = writer.episode_video_paths(episode_index)
                main_abs = Path(args.data_dir) / main_rel
                wrist_abs = Path(args.data_dir) / wrist_rel
                main_abs.parent.mkdir(parents=True, exist_ok=True)
                wrist_abs.parent.mkdir(parents=True, exist_ok=True)
                vw_main = cv2.VideoWriter(str(main_abs), fourcc, args.video_fps, (args.width, args.height))
                vw_wrist = cv2.VideoWriter(str(wrist_abs), fourcc, args.video_fps, (args.width, args.height))

                states: list[np.ndarray] = []
                actions: list[np.ndarray] = []
                timestamps: list[float] = []
                success = False
                t0 = time.perf_counter()

                for _ in range(task.max_steps):
                    if cmd["quit"] or cmd["save"] or cmd["discard"]:
                        break
                    action = build_action(keys, args.pos_step, args.rot_step, args.gripper_step)
                    out = env.step(action=action, sim_steps_per_control=sim_steps_per_control, task=task)
                    states.append(out.state)
                    actions.append(out.action)
                    timestamps.append(time.perf_counter() - t0)
                    success = out.success

                    rgb_main = env.render_main(width=args.width, height=args.height)
                    rgb_wrist = env.render_wrist(width=args.width, height=args.height)
                    vw_main.write(cv2.cvtColor(rgb_main, cv2.COLOR_RGB2BGR))
                    vw_wrist.write(cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR))
                    v.sync()
                    if out.done:
                        break

                vw_main.release()
                vw_wrist.release()

                if cmd["discard"] or len(states) == 0:
                    main_abs.unlink(missing_ok=True)
                    wrist_abs.unlink(missing_ok=True)
                    cmd["discard"] = False
                    episode_index += 1
                    continue

                writer.add_episode(
                    EpisodeRecord(
                        episode_index=episode_index,
                        task_index=0,
                        task=task.instruction,
                        state=states,
                        action=actions,
                        timestamp=timestamps,
                        success=success,
                        video_main_relpath=main_rel,
                        video_wrist_relpath=wrist_rel,
                    )
                )
                print(f"[saved] ep={episode_index}, task={task.task_id}, steps={len(states)}, success={success}")
                saved += 1
                episode_index += 1
                cmd["save"] = False

    writer.finalize()
    print(f"[done] dataset saved at {args.data_dir}")


if __name__ == "__main__":
    main()
