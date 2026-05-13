#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
import os
from PIL import Image
from core.my_env import MyEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Layout randomness: keep SEED=None and call `reset()` with no args between episodes
# so NumPy's RNG advances and cube pose / target position change each time.
# Pass an int to my_env(..., seed=K) only when you need a reproducible *first* scene;
# do not pass seed into every `reset()` during collection, or you repeat the same layout.


REPO_NAME = "ningyv/auboI10"
NUM_DEMO = 10  # Number of demonstrations to collect
ROOT = "/Users/ningyu/code_before_paper/MyI10Tele/data"  # The root directory to save the demonstrations


I10_path = "/Users/ningyu/code_before_paper/MyI10Tele/assets/aubo_i10_2/aubo_i10.xml"
import mujoco

model = mujoco.MjModel.from_xml_path(I10_path)
print(model.body_pos)


TASK_NAME = "Put cube on the black platform"
xml_path = "/Users/ningyu/code_before_paper/MyI10Tele/assets/aubo_i10_inspire/myscene.xml"
# xml_path = './asset/example_scene_y_i10.xml'
# Define the environment
PnPEnv = MyEnv(xml_path, seed=42)
print(f"action_type: {PnPEnv.action_type}")
print(f"state_type: {PnPEnv.state_type}")


create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == "y":
        import shutil

        shutil.rmtree(ROOT)
    else:
        create_new = False


if create_new:
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=ROOT,
        robot_type="aubo_i10_inspire",
        fps=20,  # 20 frames per second
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
                # "shape": (7 if PnPEnv.state_type == 'qpos' else 6,),
                "shape": (7,),
                "names": ["state"],  # 6 joint angles and 1 gripper ////  x, y, z, roll, pitch, yaw
            },
            "action": {
                "dtype": "float32",
                # "shape": (6 if PnPEnv.action_type == 'ee_pose' else 7,),
                "shape": (7,),
                "names": ["action"],  # x, y, z, roll, pitch, yaw /// 6 joint angles and 1 gripper
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["obj_init"],  # just the initial position of the object. Not used in training.
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)


action = np.zeros(7)
episode_id = 0
record_flag = False  # Start recording when the robot starts moving
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # check if the episode is done
        done = PnPEnv.check_success()
        if done:
            # Save the episode data and reset the environment
            dataset.save_episode()
            PnPEnv.reset()
            episode_id += 1
            record_flag = False
        # Teleoperate the robot and get delta end-effector pose with gripper
        action, reset = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            # Reset the environment and clear the episode buffer
            # This can be done by pressing 'z' key
            PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # Step the environment
        # Get the end-effector pose and images
        # obs_action = PnPEnv.get_ee_pose()
        obs_action = PnPEnv.get_obs_action()
        # assert obs_action.type == PnPEnv.action_type , print(f"expect action_type: {PnPEnv.action_type}, but got {obs_action.type}")
        assert obs_action.type == "qpos"
        agent_image, wrist_image = PnPEnv.grab_image()
        # # resize to 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        obs_state = PnPEnv.step(action)

        # from IPython.display import display, clear_output
        # clear_output(wait=True)
        # print(f"gripper_qpos: {PnPEnv.env.get_qpos_joint('rh_r1')}") # close : 0.81454458 open :2.7e-6
        # print(f"gripper_qpos: {PnPEnv.env.get_qpos_joint('rh_r1')[0]}")

        assert obs_state.type == PnPEnv.state_type, f"expect state_type: {PnPEnv.state_type}, but got {obs_state.type}"
        if record_flag:
            # Add the frame to the dataset
            dataset.add_frame(
                {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": obs_state,
                    "action": obs_action,
                    "obj_init": PnPEnv.obj_init_pose,
                    "task": TASK_NAME,
                }
            )
            print(PnPEnv.obj_init_pose)
        PnPEnv.render(teleop=True)
PnPEnv.env.close_viewer()
dataset.stop_image_writer()
dataset.finalize()
