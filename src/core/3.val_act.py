#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # ACT 策略val


import os
from re import M
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.utils import dataset_to_policy_features


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NVIDIA_VISIBLE_DEVICE"] = "0"
os.environ["DISPLAY"] = ":11.0"
device = torch.device("cuda:0")


dataset_metadata = LeRobotDatasetMetadata("auboI10", root="/home/ningyu/MyI10Tele/data2/")
total_episodes = dataset_metadata.total_episodes
print(f"total_episodes: {total_episodes}")

features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# Keep all dataset inputs (including observation.wrist_image) for multi-view ACT.

# Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
# we'll just use the defaults and so no arguments other than input/output features need to be passed.
# Load policy weights on CPU first so CUDA is not initialized before GLFW/MuJoCo create the GL
# context. Initializing CUDA first can crash native rendering (e.g. settexture in libmujoco) on some drivers.
cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
    chunk_size=100,
    n_action_steps=1,
    temporal_ensemble_coeff=0.9,
    dropout=0.1,
    device="cpu",
)
policy = ACTPolicy.from_pretrained("./.ckpt/auboI10_act_w_2_view_temporal_ensemble_coeff09", config=cfg, dataset_stats=dataset_metadata.stats)

# cfg = ACTConfig(input_features=input_features, output_features=output_features)
# policy = ACTPolicy.from_pretrained("./.ckpt/auboI10_act_w_2_view",config=cfg,dataset_stats=dataset_metadata.stats)

# This allows us to construct the data with action chunking
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)


# If you want to randomize the object positions, set this to None
# If you fix the seed, the object positions will be the same every time
# SEED = None <- Uncomment this line to randomize the object positions

REPO_NAME = "auboI10"
# ROOT = "/Users/ningyu/code_before_paper/MyI10Tele/data" # The root directory to save the demonstrations
ROOT = "/home/ningyu/MyI10Tele/data2"  # The root directory to save the demonstrations


from core.my_env import MyEnv

TASK_NAME = "Put cube on the black platform"
# xml_path = '/Users/ningyu/code_before_paper/MyI10Tele/assets/aubo_i10_inspire/myscene.xml'
xml_path = "/home/ningyu/MyI10Tele/assets/aubo_i10_inspire/myscene.xml"
# xml_path = './asset/example_scene_y_i10.xml'
# Define the environment

PnPEnv = MyEnv(xml_path, seed=42, action_type="qpos", state_type="qpos")
print(f"action_type: {PnPEnv.action_type}")
print(f"state_type: {PnPEnv.state_type}")

policy.to(device)


import torch
import torchvision.transforms as T
import numpy as np

# 评估参数设置
num_episodes = 20  # 测试的总轮次
max_steps_per_episode = 600  # 每轮最大步数，防止失败时陷入死循环
successful_episodes = 0  # 记录成功的次数

policy.eval()

# 优化图像预处理流程，使用 Compose 合并操作
img_transform = T.Compose([T.ToPILImage(), T.Resize((256, 256)), T.ToTensor()])

print(f"开始评估，共计 {num_episodes} 轮...")

for episode in range(num_episodes):
    # 重置环境和策略状态，使用 episode 作为 seed 保证每次测试初始状态不同
    PnPEnv.reset()
    policy.reset()

    step = 0
    episode_success = False

    # 增加 max_steps 限制，并确保可视化窗口仍在运行
    while PnPEnv.env.is_viewer_alive() and step < max_steps_per_episode:
        PnPEnv.step_env()

        if PnPEnv.env.loop_every(HZ=20):
            # 1. Observation must match training data: teleop stores observation.state from step()
            # after teleop, which for state_type=="ee_pose" is flange pose (xyz+rpy+gripper).
            # Action labels in the dataset are joint-space rows from get_obs_action(); policy predicts those.
            obs = PnPEnv.get_joint_state()
            agent_img, wrist_img = PnPEnv.grab_image()

            # 2. 图像与张量预处理
            # 使用前面定义的 transform，并通过 unsqueeze 增加 batch 维度
            image_tensor = img_transform(agent_img).unsqueeze(0).to(device)
            wrist_tensor = img_transform(wrist_img).unsqueeze(0).to(device)

            state_tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0).to(device)
            timestamp_tensor = torch.tensor([step / 20.0], dtype=torch.float32).to(device)

            data = {
                "observation.state": state_tensor,
                "observation.image": image_tensor,
                "observation.wrist_image": wrist_tensor,
                "task": ["Put cube on the black platform"],
                "timestamp": timestamp_tensor,
            }

            # 3. 策略推理 (加入 no_grad 节省显存和加速)
            with torch.no_grad():
                action = policy.select_action(data)

            action_np = action[0].cpu().numpy()

            # 4. 在环境中执行动作
            _ = PnPEnv.step(action_np)
            PnPEnv.render()
            step += 1

            # 5. 检查是否成功
            if PnPEnv.check_success():
                print(f"第 {episode + 1} 轮: 成功! (耗时 {step} 步)")
                episode_success = True
                successful_episodes += 1
                break  # 成功后跳出当前轮次的循环

    # 如果达到最大步数仍未成功
    if not episode_success and PnPEnv.env.is_viewer_alive():
        print(f"第 {episode + 1} 轮: 失败 (达到最大步数 {max_steps_per_episode})")

    # 如果用户手动关闭了渲染窗口，则提前终止评估
    if not PnPEnv.env.is_viewer_alive():
        print("渲染窗口已关闭，提前终止评估。")
        break

# 6. 计算并打印最终成功率统计
total_evaluated = episode + 1 if not PnPEnv.env.is_viewer_alive() else num_episodes
success_rate = (successful_episodes / total_evaluated) * 100 if total_evaluated > 0 else 0.0

print("-" * 30)
print("评估完成!")
print(f"总计测试轮次: {total_evaluated}")
print(f"成功轮次: {successful_episodes}")
print(f"成功率: {success_rate:.2f}%")
print("-" * 30)
