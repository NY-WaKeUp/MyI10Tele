#!/usr/bin/env python3
"""Direct ee absolute-command tracking smoke test for MyEnv.

This bypasses policy inference and sends deterministic absolute ee commands.
If executed ee motion is consistently much smaller than commanded motion, the
bottleneck is in IK/control execution rather than policy output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.dataset_config import XML_PATH
from core.my_env import MyEnv


def wrap_pi(angles: np.ndarray) -> np.ndarray:
    return ((np.asarray(angles, dtype=np.float64) + np.pi) % (2 * np.pi)) - np.pi


def pose_delta_6d(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64)[:6] - np.asarray(ref, dtype=np.float64)[:6]
    delta[3:6] = wrap_pi(delta[3:6])
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120, help="Number of control steps.")
    parser.add_argument("--dx", type=float, default=0.02, help="Absolute target x offset in meters.")
    parser.add_argument("--dy", type=float, default=0.0, help="Absolute target y offset in meters.")
    parser.add_argument("--dz", type=float, default=0.0, help="Absolute target z offset in meters.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("hold-offset", "progressive-x"),
        default="hold-offset",
        help="hold-offset: keep target at current+offset; progressive-x: add dx each step.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=50,
        help="Physics steps to hold each absolute command before measuring ee_post.",
    )
    parser.add_argument("--teleop-render", action="store_true", help="Enable overlay rendering.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("~/MyI10Tele/ee_absolute_tracking_smoke.json").expanduser(),
        help="Path to save summary json.",
    )
    args = parser.parse_args()

    env = MyEnv(
        XML_PATH,
        seed=args.seed,
        action_type="ee_pose",
        state_type="qpos",
        ee_pose_command="absolute",
    )
    print(f"action_type={env.action_type} state_type={env.state_type} ee_pose_command={env.ee_pose_command}")

    # Warm up one control tick to stabilize bookkeeping.
    env.step_env()
    if args.teleop_render:
        env.grab_image()
        env.render(teleop=True)

    records = []
    base_target = None
    for step in range(args.steps):
        env.step_env()

        ee_pre = np.asarray(env.get_ee_pose(), dtype=np.float64)
        cmd = ee_pre.copy()

        if args.mode == "hold-offset":
            if base_target is None:
                base_target = ee_pre.copy()
                base_target[0] += args.dx
                base_target[1] += args.dy
                base_target[2] += args.dz
            cmd = base_target.copy()
            cmd[3:6] = ee_pre[3:6]
            cmd[6] = ee_pre[6]
        else:
            cmd[0] = ee_pre[0] + args.dx
            cmd[1] = ee_pre[1] + args.dy
            cmd[2] = ee_pre[2] + args.dz
            cmd[3:6] = ee_pre[3:6]
            cmd[6] = ee_pre[6]

        env.step(cmd.astype(np.float64))
        for _ in range(max(1, args.settle_steps)):
            env.step_env()
        ee_post = np.asarray(env.get_ee_pose(), dtype=np.float64)
        if args.teleop_render:
            env.grab_image()
            env.render(teleop=True)

        cmd_delta = pose_delta_6d(cmd, ee_pre)
        exec_delta = pose_delta_6d(ee_post, ee_pre)
        rec = {
            "step": int(step),
            "cmd_xyz_norm": float(np.linalg.norm(cmd_delta[:3])),
            "exec_xyz_norm": float(np.linalg.norm(exec_delta[:3])),
            "cmd_6d_norm": float(np.linalg.norm(cmd_delta)),
            "exec_6d_norm": float(np.linalg.norm(exec_delta)),
            "ik_err": float(getattr(env, "last_ik_err", 0.0)),
            "efficiency_xyz": float(
                np.linalg.norm(exec_delta[:3]) / max(np.linalg.norm(cmd_delta[:3]), 1e-9)
            ),
        }
        records.append(rec)

    if not records:
        raise ValueError("No records collected. Increase --steps.")

    cmd_xyz = np.array([r["cmd_xyz_norm"] for r in records], dtype=np.float64)
    exec_xyz = np.array([r["exec_xyz_norm"] for r in records], dtype=np.float64)
    eff_xyz = np.array([r["efficiency_xyz"] for r in records], dtype=np.float64)
    cmd_6d = np.array([r["cmd_6d_norm"] for r in records], dtype=np.float64)
    exec_6d = np.array([r["exec_6d_norm"] for r in records], dtype=np.float64)
    ik_err = np.array([r["ik_err"] for r in records], dtype=np.float64)

    summary = {
        "steps_recorded": int(len(records)),
        "mode": args.mode,
        "settle_steps": int(args.settle_steps),
        "command_offset_xyz_m": [args.dx, args.dy, args.dz],
        "cmd_xyz_mean": float(cmd_xyz.mean()),
        "exec_xyz_mean": float(exec_xyz.mean()),
        "efficiency_xyz_mean": float(eff_xyz.mean()),
        "cmd_6d_mean": float(cmd_6d.mean()),
        "exec_6d_mean": float(exec_6d.mean()),
        "exec_xyz_p90": float(np.quantile(exec_xyz, 0.9)),
        "efficiency_xyz_p90": float(np.quantile(eff_xyz, 0.9)),
        "ik_err_mean": float(ik_err.mean()),
        "ik_err_p90": float(np.quantile(ik_err, 0.9)),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")

    print("\n=== EE Absolute Tracking Smoke ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()

