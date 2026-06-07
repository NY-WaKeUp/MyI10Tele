#!/usr/bin/env python3
"""Sweep arm position actuator kp/kv (+ optional physics settle) for qpos tracking.

Bypasses policy inference. Uses dataset GT qpos targets or synthetic joint deltas.
Goal: find gains where track_err/cmd drops without knocking the cube off the table.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from core.dataset_config import AUBOI10_QPOS_ROOT_CONTINUOUS, XML_PATH
from core.eval_action_guard import cube_on_table, cube_pose
from core.my_env import MyEnv, openpi_gripper_to_rh_r1_ctrl


def _to_numpy7(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    return arr[:7].astype(np.float64, copy=False)


def load_gt_episode(lerobot_root: str, episode_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    root = Path(os.path.expanduser(lerobot_root))
    parquets = sorted((root / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet under {root / 'data'}")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    ep = df.loc[df["episode_index"] == episode_id].sort_values("frame_index")
    if ep.empty:
        raise ValueError(f"Episode {episode_id} empty in {root}")
    states = np.stack([_to_numpy7(v) for v in ep["observation.state"].values])
    actions = np.stack([_to_numpy7(v) for v in ep["actions"].values])
    obj_init = None
    if "obj_init" in ep.columns:
        obj_init = np.asarray(ep.iloc[0]["obj_init"], dtype=np.float64).reshape(-1)
    return states, actions, obj_init


def snap_robot_to_state(env: MyEnv, state7: np.ndarray) -> None:
    import mujoco

    q_arm = np.asarray(state7[:6], dtype=np.float64)
    env.env.forward(q=q_arm, joint_names=env.joint_names, increase_tick=False)
    grip = openpi_gripper_to_rh_r1_ctrl(float(state7[6]))
    env.q = np.concatenate([q_arm, np.array([grip], dtype=np.float64)])
    env.compute_q = q_arm.copy()
    env.last_q = q_arm.copy()
    mujoco.mj_forward(env.env.model, env.env.data)


def run_trial(
    env: MyEnv,
    actions: np.ndarray,
    *,
    kp: float,
    kv: float,
    settle_steps: int,
    states0: np.ndarray | None,
    obj_init: np.ndarray | None,
    seed: int,
) -> dict:
    env.set_arm_position_gains(kp, kv)
    if obj_init is not None:
        env.reset_with_recorded_layout(obj_init, seed=seed)
    else:
        env.reset(seed=seed)
    if states0 is not None:
        snap_robot_to_state(env, states0[0])
    for _ in range(10):
        env.step_env()

    cube_init = cube_pose(env)
    track_errs: list[float] = []
    cmd_norms: list[float] = []
    cube_fell_step: int | None = None

    for step, action_np in enumerate(actions):
        env.step_env()
        q_pre = np.asarray(env.get_joint_state(), dtype=np.float64)
        cmd_norms.append(float(np.linalg.norm(action_np[:6] - q_pre[:6])))
        env.step(action_np)
        for _ in range(max(1, settle_steps)):
            env.step_env()
        q_post = np.asarray(env.get_joint_state(), dtype=np.float64)
        track_errs.append(float(np.linalg.norm(action_np[:6] - q_post[:6])))
        if cube_fell_step is None and not cube_on_table(cube_pose(env)):
            cube_fell_step = step

    track = np.asarray(track_errs, dtype=np.float64)
    cmd = np.asarray(cmd_norms, dtype=np.float64)
    ratio = track / np.maximum(cmd, 1e-9)
    cube_final = cube_pose(env)
    cube_disp = float(np.linalg.norm(cube_final - cube_init))

    return {
        "kp": kp,
        "kv": kv,
        "settle_steps": settle_steps,
        "steps": int(len(actions)),
        "cmd_mean": float(cmd.mean()),
        "track_mean": float(track.mean()),
        "track_cmd_ratio_mean": float(ratio.mean()),
        "track_cmd_ratio_p90": float(np.quantile(ratio, 0.9)),
        "efficiency_mean": float((1.0 - ratio).mean()),
        "cube_disp_m": cube_disp,
        "cube_on_table_end": bool(cube_on_table(cube_final)),
        "cube_fell_step": cube_fell_step,
    }


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=str, default=AUBOI10_QPOS_ROOT_CONTINUOUS)
    parser.add_argument("--gt-episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120, help="Control ticks per trial.")
    parser.add_argument(
        "--source",
        choices=("gt", "synthetic"),
        default="gt",
        help="gt: dataset actions; synthetic: q_pre + fixed joint-2 delta.",
    )
    parser.add_argument("--synthetic-dq", type=float, default=0.012, help="Rad/step for synthetic mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kp-list", type=str, default="2000,5000,10000")
    parser.add_argument("--kv-list", type=str, default="200,500,1000")
    parser.add_argument("--settle-list", type=str, default="1,5,10")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("~/MyI10Tele/qpos_tracking_sweep.json").expanduser(),
    )
    args = parser.parse_args()

    kp_list = parse_float_list(args.kp_list)
    kv_list = parse_float_list(args.kv_list)
    settle_list = parse_int_list(args.settle_list)

    states, gt_actions, obj_init = load_gt_episode(args.lerobot_root, args.gt_episode)
    gt_actions = gt_actions[: args.max_steps]
    states = states[: args.max_steps]

    if args.source == "synthetic":
        actions = []
        q = states[0].copy()
        for _ in range(args.max_steps):
            target = q.copy()
            target[1] += args.synthetic_dq
            actions.append(target)
            q = target
        actions_np = np.stack(actions)
        states0 = None
        obj_init = None
    else:
        actions_np = gt_actions
        states0 = states

    env = MyEnv(
        XML_PATH,
        seed=args.seed,
        action_type="qpos",
        state_type="qpos",
        ee_pose_command="absolute",
    )
    default_kp, default_kv = env.get_arm_position_gains()
    print(f"XML default arm gains: kp={default_kp} kv={default_kv}")
    print(
        f"Sweep: {len(kp_list)}x{len(kv_list)}x{len(settle_list)} configs, "
        f"{len(actions_np)} steps, source={args.source}"
    )

    results: list[dict] = []
    for kp, kv, settle in product(kp_list, kv_list, settle_list):
        rec = run_trial(
            env,
            actions_np,
            kp=kp,
            kv=kv,
            settle_steps=settle,
            states0=states0,
            obj_init=obj_init if args.source == "gt" else None,
            seed=args.seed,
        )
        results.append(rec)
        flag = "OK" if rec["cube_on_table_end"] else "FELL"
        print(
            f"kp={kp:6.0f} kv={kv:5.0f} settle={settle:2d} "
            f"track/cmd={rec['track_cmd_ratio_mean']:.3f} "
            f"track={rec['track_mean']:.4f} cmd={rec['cmd_mean']:.4f} "
            f"cube_disp={rec['cube_disp_m']*100:.1f}cm [{flag}]",
            flush=True,
        )

    stable = [r for r in results if r["cube_on_table_end"]]
    stable.sort(key=lambda r: (r["track_cmd_ratio_mean"], r["track_mean"]))
    best = stable[0] if stable else min(results, key=lambda r: r["track_cmd_ratio_mean"])

    summary = {
        "source": args.source,
        "gt_episode": args.gt_episode,
        "max_steps": args.max_steps,
        "xml_default_kp": default_kp,
        "xml_default_kv": default_kv,
        "best_stable": best,
        "best_any": min(results, key=lambda r: r["track_cmd_ratio_mean"]),
        "n_configs": len(results),
        "n_stable": len(stable),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Qpos Tracking Sweep ===")
    print(f"Best (cube on table): {best}")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
