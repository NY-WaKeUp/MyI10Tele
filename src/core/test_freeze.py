import sys
import numpy as np
from PIL import Image
from core.my_env import MyEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import shutil

SEED = 0
REPO_NAME = "auboI10"
NUM_DEMO = 2
ROOT = "/Users/ningyu/code_before_paper/MyI10Tele/data_test"

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

dataset = LeRobotDataset.create(
    repo_id=REPO_NAME,
    root=ROOT,
    robot_type="aubo_i10_inspire",
    fps=20,  # 20 frames per second
    features={
        "observation.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        "observation.wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
        "obj_init": {"dtype": "float32", "shape": (6,), "names": ["obj_init"]},
    },
    image_writer_threads=10,
    image_writer_processes=5,
)

TASK_NAME = "Put cube on the black platform"
xml_path = "/Users/ningyu/code_before_paper/MyI10Tele/assets/aubo_i10_inspire/myscene.xml"
PnPEnv = MyEnv(xml_path, seed=SEED, state_type="joint_angle")

action = np.zeros(7)
episode_id = 0
record_flag = False
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        done = PnPEnv.check_success()
        if done:
            print(f"Episode {episode_id} done! Saving...")
            dataset.save_episode()
            print("Resetting Env")
            PnPEnv.reset(seed=SEED)
            episode_id += 1
            print(f"Current episode_id = {episode_id}")
            record_flag = False  # Need to add this?

        action, reset = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            print("Reset requested by user!")
            PnPEnv.reset(seed=SEED)
            dataset.clear_episode_buffer()
            record_flag = False

        ee_pose = PnPEnv.get_ee_pose()
        agent_image, wrist_image = PnPEnv.grab_image()

        agent_image = Image.fromarray(agent_image).resize((256, 256))
        wrist_image = Image.fromarray(wrist_image).resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)
        if record_flag:
            dataset.add_frame(
                {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose,
                    "action": joint_q,
                    "obj_init": PnPEnv.obj_init_pose,
                    "task": TASK_NAME,
                }
            )
        PnPEnv.render(teleop=True)

print("Exited main loop!")
if episode_id >= NUM_DEMO:
    print("Reached NUM_DEMO limit. Consolidating dataset...")
    dataset.consolidate()
