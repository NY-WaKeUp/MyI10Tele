#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # ACT 策略val


import os
from re import M
from core.my_policy import MyPolicy
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

MyPolicy.set_visible_cuda_devices("0")
os.environ["DISPLAY"] = ":11.0"
device = torch.device("cuda:0")


from core.dataset_config import (
    ACTION_LABEL,
    REPO_NAME as DATASET_REPO,
    TASK_NAME,
    XML_PATH,
    dataset_root,
    env_action_type,
)

DATASET_ROOT = dataset_root()

dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
input_features, output_features = MyPolicy.input_output_features_from_metadata(
    dataset_metadata
)

cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
    chunk_size=100,
    n_action_steps=1,
    temporal_ensemble_coeff=0.9,
    dropout=0.1,
    device="cpu",
)
pretrained_model_id = "lerobot/act_aloha_sim_transfer_cube_human"
save_dir = f".ckpt/{pretrained_model_id.split('/')[-1]}_auboI10_{ACTION_LABEL}"

policy = ACTPolicy.from_pretrained(
    "./.ckpt/auboI10_act_w_2_view_temporal_ensemble_coeff09",
    config=cfg,
    dataset_stats=dataset_metadata.stats,
)

# cfg = ACTConfig(input_features=input_features, output_features=output_features)
# policy = ACTPolicy.from_pretrained("./.ckpt/auboI10_act_w_2_view",config=cfg,dataset_stats=dataset_metadata.stats)

# This allows us to construct the data with action chunking
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)


# If you want to randomize the object positions, set this to None
# If you fix the seed, the object positions will be the same every time
# SEED = None <- Uncomment this line to randomize the object positions

from core.my_env import MyEnv

_eval_action = env_action_type()
PnPEnv = MyEnv(XML_PATH, seed=42, action_type=_eval_action, state_type="qpos")
print(f"dataset: {DATASET_ROOT}")
print(f"ACTION_LABEL: {ACTION_LABEL}")
print(f"action_type: {PnPEnv.action_type}")
print(f"state_type: {PnPEnv.state_type}")

policy.to(device)


import torch
import torchvision.transforms as T
import numpy as np
from core.episode_video_recorder import EpisodeVideoRecorder

# 评估参数设置
num_episodes = 20  # 测试的总轮次
max_steps_per_episode = 600  # 每轮最大步数，防止失败时陷入死循环
successful_episodes = 0  # 记录成功的次数

policy.eval()

# 优化图像预处理流程，使用 Compose 合并操作
img_transform = T.Compose([T.ToPILImage(), T.Resize((256, 256)), T.ToTensor()])

# Initialize video recorder
video_recorder = EpisodeVideoRecorder(
    output_dir="./episode_videos_act",
    fps=20,
    frame_size=(512, 256),
)

print(f"开始评估，共计 {num_episodes} 轮...")

for episode in range(num_episodes):
    # 重置环境和策略状态，使用 episode 作为 seed 保证每次测试初始状态不同
    PnPEnv.reset()
    policy.reset()

    step = 0
    episode_success = False

    # Start video recording for this episode
    video_recorder.start_episode(episode)

    # 增加 max_steps 限制，并确保可视化窗口仍在运行
    while PnPEnv.env.is_viewer_alive() and step < max_steps_per_episode:
        PnPEnv.step_env()

        if PnPEnv.env.loop_every(HZ=20):
            # observation.state = pre-step joint qpos (same for qpos/ee_pose datasets)
            obs = PnPEnv.get_joint_state()
            agent_img, wrist_img = PnPEnv.grab_image()

            # Record frame to video
            video_recorder.record_frame(agent_img, wrist_img)

            # 2. 图像与张量预处理
            # 使用前面定义的 transform，并通过 unsqueeze 增加 batch 维度
            image_tensor = img_transform(agent_img).unsqueeze(0).to(device)
            wrist_tensor = img_transform(wrist_img).unsqueeze(0).to(device)

            state_tensor = (
                torch.as_tensor(np.asarray(obs), dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            timestamp_tensor = torch.tensor([step / 20.0], dtype=torch.float32).to(
                device
            )

            data = {
                "observation.state": state_tensor,
                "observation.image": image_tensor,
                "observation.wrist_image": wrist_tensor,
                "task": [TASK_NAME],
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

    # Stop video recording and mark outcome
    video_recorder.stop(success=episode_success)

    # 如果达到最大步数仍未成功
    if not episode_success and PnPEnv.env.is_viewer_alive():
        print(f"第 {episode + 1} 轮: 失败 (达到最大步数 {max_steps_per_episode})")

    # 如果用户手动关闭了渲染窗口，则提前终止评估
    if not PnPEnv.env.is_viewer_alive():
        print("渲染窗口已关闭，提前终止评估。")
        break

# 6. 计算并打印最终成功率统计
total_evaluated = episode + 1 if not PnPEnv.env.is_viewer_alive() else num_episodes
success_rate = (
    (successful_episodes / total_evaluated) * 100 if total_evaluated > 0 else 0.0
)

print("-" * 30)
print("评估完成!")
print(f"总计测试轮次: {total_evaluated}")
print(f"成功轮次: {successful_episodes}")
print(f"成功率: {success_rate:.2f}%")
print(f"视频已保存到: ./episode_videos_act/")
print("-" * 30)
