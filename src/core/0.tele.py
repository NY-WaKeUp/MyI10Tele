#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
import numpy as np
import cv2

from core.my_env import MyEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_NAME = "ningyv/auboI10"
NUM_DEMO = 10
# ROOT = "/Users/ningyu/code_before_paper/MyI10Tele/data_w_shadow_x264"
ROOT = "/Users/ningyu/code_before_paper/MyI10Tele/data_w_shadow_h264_znear0001"
# ROOT = "/Users/ningyu/code_before_paper/MyI10Tele/data_w_shadow_h264_znear0001_fov179"

TASK_NAME = "Put cube on the black platform"
XML_PATH = (
    "/Users/ningyu/code_before_paper/MyI10Tele/assets/aubo_i10_inspire/myscene.xml"
)


def main() -> None:
    # macOS spawn + GLFW: subprocess image writers re-import this file; keep processes off.
    image_writer_processes = 0 if sys.platform == "darwin" else 5

    pn_env = MyEnv(XML_PATH, seed=42)
    print(f"action_type: {pn_env.action_type}")
    print(f"state_type: {pn_env.state_type}")

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
            fps=20,
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
        if pn_env.env.loop_every(HZ=20):
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
            obs_actions = pn_env.get_obs_action()
            assert obs_actions.type == "qpos"
            agent_image, wrist_image = pn_env.grab_image()
            agent_image = cv2.resize(
                agent_image, (256, 256), interpolation=cv2.INTER_AREA
            )
            wrist_image = cv2.resize(
                wrist_image, (256, 256), interpolation=cv2.INTER_AREA
            )
            obs_state = pn_env.step(actions)
            assert (
                obs_state.type == pn_env.state_type
            ), f"expect state_type: {pn_env.state_type}, but got {obs_state.type}"
            if record_flag:
                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": obs_state,
                        "actions": obs_actions,
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
