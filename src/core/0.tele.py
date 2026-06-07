#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
from pathlib import Path

# Allow `python 0.tele.py` from src/core without PYTHONPATH=src.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import cv2

from core.my_env import MyEnv
from core.dataset_config import (
    REPO_NAME,
    TASK_NAME,
    XML_PATH,
    teleop_ee_pose_root,
    teleop_qpos_root,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT_QPOS = teleop_qpos_root()
ROOT_EE = teleop_ee_pose_root()
NUM_DEMO = 50
# Keep loop rate and LeRobot fps identical (pi0 / ACT assume dataset timestamps match this).
HZ = 20
# Skip frames where IK barely moved the arm; still log gripper toggles.
MIN_ARM_DQ_RAD = 1e-4
MIN_GRIPPER_DQ = 0.05

DATASET_FEATURES = {
    "observation.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["state"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],
    },
    "obj_init": {
        "dtype": "float32",
        # cube_xyz(3) + cube_quat_wxyz(4) + platform_xyz(3)
        "shape": (10,),
        "names": ["obj_init"],
    },
}


def _create_or_load_dataset(root: str, create_new: bool) -> LeRobotDataset:
    image_writer_processes = 0 if sys.platform == "darwin" else 5
    _vcodec = "auto" if sys.platform == "darwin" else "h264"
    if create_new:
        return LeRobotDataset.create(
            repo_id=REPO_NAME,
            root=root,
            robot_type="aubo_i10_inspire",
            fps=HZ,
            vcodec=_vcodec,
            streaming_encoding=True,
            encoder_queue_maxsize=90,
            features=DATASET_FEATURES,
            image_writer_threads=6,
            image_writer_processes=image_writer_processes,
        )
    print(f"Load from previous dataset: {root}")
    return LeRobotDataset(REPO_NAME, root=root)


def main() -> None:
    # Keyboard teleop uses ee deltas; both qpos and ee_pose action datasets are written.
    pn_env = MyEnv(
        XML_PATH,
        seed=42,
        action_type="ee_pose",
        state_type="qpos",
        ee_pose_command="delta",
    )
    print(f"action_type: {pn_env.action_type}")
    print(f"state_type: {pn_env.state_type}")
    print(f"ee_pose_command: {pn_env.ee_pose_command}")
    print(f"qpos dataset root: {ROOT_QPOS}")
    print(f"ee_pose dataset root: {ROOT_EE}")

    roots = (ROOT_QPOS, ROOT_EE)
    existing = [r for r in roots if os.path.exists(r)]
    create_new = True
    if existing:
        print("Existing dataset dirs:")
        for r in existing:
            print(f"  {r}")
        ans = input("Delete all and start fresh? (y/n) ")
        if ans == "y":
            for r in existing:
                shutil.rmtree(r)
        else:
            create_new = False

    dataset_qpos = _create_or_load_dataset(ROOT_QPOS, create_new)
    dataset_ee = _create_or_load_dataset(ROOT_EE, create_new)

    actions = np.zeros(7)
    episode_id = 0
    record_flag = False
    while pn_env.env.is_viewer_alive() and episode_id < NUM_DEMO:
        pn_env.step_env()
        if pn_env.env.loop_every(HZ=HZ):
            done = pn_env.check_success()
            if done:
                # macOS: avoid ProcessPoolExecutor in save_episode (spawn + GLFW); streaming path is already fast.
                parallel = sys.platform != "darwin"
                dataset_qpos.save_episode(parallel_encoding=parallel)
                dataset_ee.save_episode(parallel_encoding=parallel)
                pn_env.reset()
                episode_id += 1
                record_flag = False
            actions, reset = pn_env.teleop_robot()
            if not record_flag and np.any(actions != 0):
                record_flag = True
                print("Start recording")
            if reset:
                pn_env.reset()
                for ds in (dataset_qpos, dataset_ee):
                    buf = getattr(ds, "episode_buffer", None)
                    if buf is not None:
                        ds.clear_episode_buffer()
                record_flag = False
                continue
            pre_state = np.array(pn_env.get_joint_state(), dtype=np.float32)
            agent_image, wrist_image = pn_env.grab_image()
            agent_image = cv2.resize(
                agent_image, (256, 256), interpolation=cv2.INTER_AREA
            )
            wrist_image = cv2.resize(
                wrist_image, (256, 256), interpolation=cv2.INTER_AREA
            )
            pn_env.step(actions)
            pn_env.step_env()
            post_q = np.array(pn_env.get_joint_state(), dtype=np.float32)
            post_ee = np.array(pn_env.get_ee_pose(), dtype=np.float32)
            dq_arm = float(np.linalg.norm(post_q[:6] - pre_state[:6]))
            dq_grip = float(abs(post_q[6] - pre_state[6]))
            # if record_flag and (dq_arm > MIN_ARM_DQ_RAD or dq_grip > MIN_GRIPPER_DQ):
            if record_flag:
                frame = {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": pre_state,
                    "obj_init": pn_env.obj_init_pose,
                    "task": TASK_NAME,
                }
                dataset_qpos.add_frame({**frame, "actions": post_q})
                dataset_ee.add_frame({**frame, "actions": post_ee})
            pn_env.render(teleop=True)

    pn_env.env.close_viewer()
    for ds in (dataset_qpos, dataset_ee):
        ds.stop_image_writer()
        ds.finalize()


if __name__ == "__main__":
    main()
