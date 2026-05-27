#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
import numpy as np
import cv2

from core.my_env import MyEnv
from core.dataset_config import (
    ACTION_LABEL,
    REPO_NAME,
    TASK_NAME,
    XML_PATH,
    dataset_root,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = dataset_root()
NUM_DEMO = 50
# Keep loop rate and LeRobot fps identical (pi0 / ACT assume dataset timestamps match this).
HZ = 20
# Skip frames where IK barely moved the arm; still log gripper toggles.
MIN_ARM_DQ_RAD = 1e-4
MIN_GRIPPER_DQ = 0.05


def main() -> None:
    # macOS spawn + GLFW: subprocess image writers re-import this file; keep processes off.
    image_writer_processes = 0 if sys.platform == "darwin" else 5

    # Keyboard teleop uses ee deltas; dataset actions column uses ACTION_LABEL (qpos or ee_pose).
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
    print(f"ACTION_LABEL (stored in dataset): {ACTION_LABEL}")
    print(f"dataset root: {ROOT}")

    create_new = True
    if os.path.exists(ROOT):
        print(f"Directory {ROOT} already exists.")
        ans = input("Do you want to delete it? (y/n) ")
        if ans == "y":
            shutil.rmtree(ROOT)
        else:
            create_new = False

    # Streaming encode during capture -> save_episode avoids PNG->AV1 batch. "auto" picks VideoToolbox on macOS.
    _vcodec = "auto" if sys.platform == "darwin" else "h264"

    if create_new:
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            root=ROOT,
            robot_type="aubo_i10_inspire",
            fps=HZ,
            vcodec=_vcodec,
            streaming_encoding=True,
            encoder_queue_maxsize=90,
            features={
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
                    "shape": (6,),
                    "names": ["obj_init"],
                },
            },
            image_writer_threads=6,
            image_writer_processes=image_writer_processes,
        )
    else:
        print("Load from previous dataset")
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)

    actions = np.zeros(7)
    episode_id = 0
    record_flag = False
    while pn_env.env.is_viewer_alive() and episode_id < NUM_DEMO:
        pn_env.step_env()
        if pn_env.env.loop_every(HZ=HZ):
            done = pn_env.check_success()
            if done:
                # macOS: avoid ProcessPoolExecutor in save_episode (spawn + GLFW); streaming path is already fast.
                dataset.save_episode(parallel_encoding=sys.platform != "darwin")
                pn_env.reset()
                episode_id += 1
                record_flag = False
            actions, reset = pn_env.teleop_robot()
            if not record_flag and np.any(actions != 0):
                record_flag = True
                print("Start recording")
            if reset:
                pn_env.reset()
                # Loaded datasets keep episode_buffer=None until first add_frame; clear_episode_buffer would crash.
                buf = getattr(dataset, "episode_buffer", None)
                if buf is not None:
                    dataset.clear_episode_buffer()
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
            if ACTION_LABEL == "qpos":
                post_action = post_q
            elif ACTION_LABEL == "ee_pose":
                post_action = np.array(pn_env.get_ee_pose(), dtype=np.float32)
            else:
                raise ValueError(f"unknown ACTION_LABEL: {ACTION_LABEL}")
            dq_arm = float(np.linalg.norm(post_q[:6] - pre_state[:6]))
            dq_grip = float(abs(post_q[6] - pre_state[6]))
            if record_flag and (dq_arm > MIN_ARM_DQ_RAD or dq_grip > MIN_GRIPPER_DQ):
                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": pre_state,
                        "actions": post_action,
                        "obj_init": pn_env.obj_init_pose,
                        "task": TASK_NAME,
                    }
                )
            pn_env.render(teleop=True)

    pn_env.env.close_viewer()
    dataset.stop_image_writer()
    dataset.finalize()


if __name__ == "__main__":
    main()
