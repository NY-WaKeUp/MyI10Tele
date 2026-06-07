#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # ACT 策略val


import os
from re import M
from core.my_policy import (
    MyPolicy,
    load_paligemma_tokenizer,
    resolve_pi0_pretrained_path,
)
from lerobot.policies.pi0.modeling_pi0 import PI0Config, PI0Policy
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

from transformers import AutoTokenizer

MyPolicy.set_visible_cuda_devices("1")
os.environ["DISPLAY"] = ":11.0"
device = torch.device("cuda:0")


DATASET_REPO = "auboI10"
# DATASET_ROOT = "/home/ningyu/MyI10Tele/data2/"
DATASET_NAME = "data_w_shadow_h264_znear0001"
DATASET_ROOT = f"/home/ningyu/MyI10Tele/{DATASET_NAME}/"
PI0_PRETRAINED_DIR: str | None = None
_allow_hub = False
_allow_hub_tokenizer = False
COMPILE_PI0_MODEL = False  #  note 和训练不一样，关掉编译
USE_GRADIENT_CHECKPOINTING = not COMPILE_PI0_MODEL
PI0_COMPILE_MODE = "default"
pretrained_model_id, pi_pretrained_local_only = resolve_pi0_pretrained_path(
    PI0_PRETRAINED_DIR,
    allow_hub_download=_allow_hub,
)

dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
total_episodes = dataset_metadata.total_episodes
print(f"total_episodes: {total_episodes}")

features = dataset_to_policy_features(dataset_metadata.features)
output_features = {
    key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

is_finetuning = True
save_dir = f".ckpt/{pretrained_model_id.split('/')[-1]}/{DATASET_NAME}"

if is_finetuning:
    print(
        f"Loading finetuned PI0 model: {save_dir} (local_files_only={pi_pretrained_local_only})"
    )
    cfg = PI0Config(
        input_features=input_features,
        output_features=output_features,
        compile_model=COMPILE_PI0_MODEL,
        compile_mode=PI0_COMPILE_MODE if COMPILE_PI0_MODEL else "default",
        dtype="bfloat16",
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        train_expert_only=True,
        # PreTrainedConfig must stay JSON/YAML-serializable (draccus); use str not torch.device.
        chunk_size=100,
        n_action_steps=1,  # ? can do this ?
        device=str(device),
    )
    policy = PI0Policy.from_pretrained(
        save_dir,
        config=cfg,
        dataset_stats=dataset_metadata.stats,
        local_files_only=pi_pretrained_local_only,
    )
    print("Train expert only for fine-tuning.")
    if COMPILE_PI0_MODEL and not USE_GRADIENT_CHECKPOINTING:
        print(
            "compile_model=True: gradient checkpointing disabled (functorch RNG functionalization "
            "with recomputed flash-attention ops is unsupported); lower batch size if CUDA OOM."
        )
    if COMPILE_PI0_MODEL:
        print(f"torch.compile mode: {PI0_COMPILE_MODE}")
        if PI0_COMPILE_MODE == "max-autotune":
            print(
                "CUDA Graphs on (max-autotune). If you see CUDAGraph overwrite errors, use "
                "PI0_CUDAGRAPHS=0 or PI0_COMPILE_MODE=max-autotune-no-cudagraphs."
            )
else:
    print("Initializing PI0 model from scratch...")
    cfg = PI0Config(
        input_features=input_features,
        output_features=output_features,
        chunk_size=100,
        n_action_steps=1,
        dtype="bfloat16",
        gradient_checkpointing=True,
        device=str(device),
    )
    policy = PI0Policy(cfg, dataset_stats=dataset_metadata.stats)

# This allows us to construct the data with action chunking
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

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

# 获取分词器路径（保持与训练一致）
tokenizer = load_paligemma_tokenizer(allow_hub_download=_allow_hub_tokenizer)
tokenizer.padding_side = "right"
TOKENIZER_MAX_LENGTH = 48  # 对应你训练脚本中的 cfg.tokenizer_max_length

# --- 推理循环内部 ---
task_string = "Put cube on the black platform"

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
    output_dir="./episode_videos_pi0",
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
            # --- 推理循环内部 ---
            task_string = "Put cube on the black platform"

            # 预处理文本
            tokens = tokenizer(
                [task_string],
                padding="max_length",
                max_length=TOKENIZER_MAX_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            data = {
                "observation.state": state_tensor,
                "observation.image": image_tensor,
                "observation.wrist_image": wrist_tensor,
                "observation.language.tokens": tokens["input_ids"],
                # 同时也建议加上 attention_mask，防止模型推理异常
                "observation.language.attention_mask": tokens["attention_mask"],
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
print(f"视频已保存到: ./episode_videos_pi0/")
print("-" * 30)
