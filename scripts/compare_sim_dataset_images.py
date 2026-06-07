#!/usr/bin/env python3
"""Compare sim camera 256×256 vs LeRobot stored frames (tele / training pixels).

For frame *t*, replays dataset actions [0..t-1] so cube/arm layout matches teleop, then
grab+resize with the same path as ``0.tele.py`` / ``openpi_obs.preprocess_lerobot_image``.

Usage (needs MuJoCo + DISPLAY)::

  cd ~/MyI10Tele
  PYTHONPATH=src python scripts/compare_sim_dataset_images.py \\
    --episode 0 --steps 0,10,50,100 \\
    --lerobot-root ~/MyI10Tele/data_auboI10_qpos_v21_continuous \\
    --out-dir ./sim_dataset_image_align_ep0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.dataset_config import AUBOI10_QPOS_ROOT_CONTINUOUS, XML_PATH, policy_ee_pose_command
from core.openpi_obs import preprocess_lerobot_image


def _parse_steps(spec: str, max_step: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted({s for s in out if 0 <= s < max_step})


def _to_numpy7(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    return arr[:7].astype(np.float64, copy=False)


def _load_gt_episode(lerobot_root: str, episode_id: int):
    import pandas as pd

    root = Path(os.path.expanduser(lerobot_root))
    with open(root / "meta" / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    num_episodes = int(info["total_episodes"])
    if not 0 <= episode_id < num_episodes:
        raise ValueError(f"episode {episode_id} not in [0, {num_episodes - 1}]")

    parquets = sorted((root / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    ep = df.loc[df["episode_index"] == episode_id].sort_values("frame_index")
    states = np.stack([_to_numpy7(v) for v in ep["observation.state"].values])
    actions = np.stack([_to_numpy7(v) for v in ep["actions"].values])
    obj_init = None
    if "obj_init" in ep.columns:
        obj_init = np.asarray(ep.iloc[0]["obj_init"], dtype=np.float64).reshape(-1)
    return states, actions, obj_init, len(ep)


def _snap_robot_to_state(env, state7: np.ndarray) -> None:
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


def _episode_indices(dataset, episode_id: int) -> np.ndarray:
    ep_table = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_id)
    return np.array(ep_table["index"], dtype=np.int64)


def _parse_lerobot_image(item: dict, key: str) -> np.ndarray:
    from openpi.policies.aubo_policy import _parse_image

    return _parse_image(item[key])


def _image_metrics(sim: np.ndarray, ds: np.ndarray) -> dict:
    a = sim.astype(np.float32)
    b = ds.astype(np.float32)
    diff = np.abs(a - b)
    mse = float((diff**2).mean())
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((255.0**2) / mse))
    return {
        "mae": float(diff.mean()),
        "rmse": float(np.sqrt(mse)),
        "max_abs": float(diff.max()),
        "psnr": psnr,
    }


def _save_panel(path: Path, ds: np.ndarray, sim: np.ndarray, title: str) -> None:
    import cv2

    diff = np.abs(sim.astype(np.int16) - ds.astype(np.int16)).astype(np.uint8)
    diff_vis = np.clip(diff.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
    panel = np.concatenate([ds, sim, diff_vis], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.putText(
        panel_bgr,
        title,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel_bgr)


def _replay_to_pre_step(env, actions: np.ndarray, target_step: int) -> None:
    for i in range(target_step):
        env.step(actions[i])
        env.step_env()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim vs LeRobot 256px image alignment")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=str,
        default="0,1,2,5,10,20,50,100,150",
        help="Comma list and/or ranges, e.g. 0,10,50-60",
    )
    parser.add_argument("--lerobot-root", type=str, default=AUBOI10_QPOS_ROOT_CONTINUOUS)
    parser.add_argument("--repo-id", type=str, default="auboI10")
    parser.add_argument("--xml-path", type=str, default=XML_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--display",
        type=str,
        default=None,
        help="X11 display (default: env DISPLAY or :51.0)",
    )
    parser.add_argument("--out-dir", type=str, default="./sim_dataset_image_align")
    parser.add_argument(
        "--snap-only",
        action="store_true",
        help="Snap arm to dataset qpos only (no action replay); cube pose will drift after early steps",
    )
    args = parser.parse_args()
    if args.display is not None:
        os.environ["DISPLAY"] = args.display
    else:
        os.environ.setdefault("DISPLAY", ":51.0")

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    from core.my_env import MyEnv

    root = os.path.expanduser(args.lerobot_root)
    states, actions, obj_init, n_frames = _load_gt_episode(root, args.episode)
    step_ids = _parse_steps(args.steps, n_frames)

    meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id, root=root)
    dataset = lerobot_dataset.LeRobotDataset(args.repo_id, root=root)
    indices = _episode_indices(dataset, args.episode)

    env = MyEnv(
        args.xml_path,
        seed=args.seed,
        action_type="qpos",
        state_type="qpos",
        ee_pose_command=policy_ee_pose_command(),
    )
    if obj_init is None:
        env.reset(seed=args.seed)
    else:
        env.reset_with_recorded_layout(obj_init, seed=args.seed)

    out_dir = Path(args.out_dir).expanduser() / f"ep{args.episode:03d}"
    rows: list[dict] = []

    for step in step_ids:
        if args.snap_only:
            _snap_robot_to_state(env, states[step])
        else:
            env.reset_with_recorded_layout(obj_init, seed=args.seed) if obj_init is not None else env.reset(seed=args.seed)
            _snap_robot_to_state(env, states[0])
            _replay_to_pre_step(env, actions, step)
            for _ in range(3):
                env.step_env()

        sim_q = np.array(env.get_joint_state(), dtype=np.float32)
        gt_q = states[step].astype(np.float32)
        qpos_err = float(np.linalg.norm(sim_q[:6] - gt_q[:6]))

        agent_raw, wrist_raw = env.grab_image()
        sim_agent = preprocess_lerobot_image(agent_raw)
        sim_wrist = preprocess_lerobot_image(wrist_raw)

        item = dataset[int(indices[step])]
        ds_agent = _parse_lerobot_image(item, "observation.image")
        ds_wrist = _parse_lerobot_image(item, "observation.wrist_image")

        m_agent = _image_metrics(sim_agent, ds_agent)
        m_wrist = _image_metrics(sim_wrist, ds_wrist)
        row = {
            "step": step,
            "qpos_l2_arm": qpos_err,
            "agent_raw_shape": list(agent_raw.shape),
            "agent": m_agent,
            "wrist": m_wrist,
            "snap_only": args.snap_only,
        }
        rows.append(row)

        tag = (
            f"ep{args.episode} s{step} agent_mae={m_agent['mae']:.2f} "
            f"wrist_mae={m_wrist['mae']:.2f} qerr={qpos_err:.4f}"
        )
        _save_panel(out_dir / f"step_{step:04d}_agent.png", ds_agent, sim_agent, tag + " | L=dataset R=sim")
        _save_panel(out_dir / f"step_{step:04d}_wrist.png", ds_wrist, sim_wrist, tag + " wrist")

        print(
            f"[step {step:4d}] qpos_err={qpos_err:.5f}  "
            f"agent MAE={m_agent['mae']:.2f} PSNR={m_agent['psnr']:.1f}  "
            f"wrist MAE={m_wrist['mae']:.2f} PSNR={m_wrist['psnr']:.1f}  "
            f"raw={agent_raw.shape}"
        )

    summary = {
        "episode": args.episode,
        "lerobot_root": root,
        "n_frames": n_frames,
        "snap_only": args.snap_only,
        "steps": rows,
        "agent_mae_mean": float(np.mean([r["agent"]["mae"] for r in rows])),
        "wrist_mae_mean": float(np.mean([r["wrist"]["mae"] for r in rows])),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("-" * 60)
    print(f"Mean agent MAE={summary['agent_mae_mean']:.2f}  wrist MAE={summary['wrist_mae_mean']:.2f}")
    print(f"Panels + summary -> {out_dir}")


if __name__ == "__main__":
    main()
