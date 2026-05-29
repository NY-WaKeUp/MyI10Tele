#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from core.dataset_config import ACTION_LABEL, REPO_NAME, dataset_root

ROOT = dataset_root(interp=True)
ACTION_DIM_LABELS = (
    ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    if ACTION_LABEL == "ee_pose"
    else [f"j{i}" for i in range(6)] + ["gripper"]
)

print(f"ACTION_LABEL: {ACTION_LABEL}")
print(f"Loading dataset from {ROOT}...")
dataset = LeRobotDataset(REPO_NAME, root=ROOT)
print(f"Total episodes: {dataset.num_episodes}")
print(f"Total frames: {dataset.num_frames}")
print(f"Average steps per episode: {dataset.num_frames / dataset.num_episodes}")


# --- Dataset Global Statistics ---
print("\n--- Dataset Global Statistics ---")
# 使用 dataset.hf_dataset.with_format("numpy") 确保获取的是 numpy 数组
hf_np = dataset.hf_dataset.with_format("numpy")
all_states = np.array(hf_np["observation.state"])
all_actions = np.array(hf_np["actions"])

# 如果数据是嵌套的 (Episode, Frame, Dim)，需要 flatten
if len(all_states.shape) > 2:
    all_states = all_states.reshape(-1, all_states.shape[-1])
    all_actions = all_actions.reshape(-1, all_actions.shape[-1])

state_min, state_max = all_states.min(axis=0), all_states.max(axis=0)
action_min, action_max = all_actions.min(axis=0), all_actions.max(axis=0)

np.set_printoptions(precision=4, suppress=True)
print(f"State Range (Min) : {state_min}")
print(f"State Range (Max) : {state_max}")
print("-" * 30)
print(f"Action Range (Min): {action_min}")
print(f"Action Range (Max): {action_max}")

# exit(0)

# --- 分析特定 Episode ---
EPISODE_ID = 0
# 正确筛选 Episode 数据
episode_data = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_ID)
num_frames_in_ep = len(episode_data)
print(f"\nAnalyzing Episode {EPISODE_ID} ({num_frames_in_ep} frames)...")

states = np.array(episode_data["observation.state"])
actions = np.array(episode_data["actions"])
actions_diff = np.diff(actions, axis=0)

# --- 2. 绘制轨迹分析图 ---
fig_plot, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for i in range(6):
    axs[0].plot(actions[:, i], label=ACTION_DIM_LABELS[i])
axs[0].set_title(f"Actions [{ACTION_LABEL}] (Episode {EPISODE_ID})")
axs[0].set_ylabel("Value")
axs[0].legend(loc="upper right", ncol=3)
axs[0].grid(True)

# 2. 夹爪同步 (Dim 6)
axs[1].plot(actions[:, 6], label="Gripper Action", linestyle="--", color="red")
axs[1].plot(states[:, 6], label="Gripper State", alpha=0.7, color="blue")
axs[1].set_title("Gripper Synchronization")
axs[1].set_ylabel("Value")
axs[1].legend(loc="upper right")
axs[1].grid(True)

# 3. 速度/平滑度
for i in range(6):
    axs[2].plot(actions_diff[:, i], alpha=0.5)
axs[2].set_title("Action Delta (Velocity Check)")
axs[2].set_xlabel("Frame")
axs[2].set_ylabel("Delta")
axs[2].grid(True)

plt.tight_layout()
# 注意：这里先不要 show()，否则会阻塞后面的交互界面

# --- 3. 高级交互回放器 ---
fig_ui = plt.figure(figsize=(14, 8))
gs = fig_ui.add_gridspec(3, 2, height_ratios=[5, 1, 0.5])
ax_agent = fig_ui.add_subplot(gs[0, 0])
ax_wrist = fig_ui.add_subplot(gs[0, 1])
ax_text = fig_ui.add_subplot(gs[1, :])
ax_slider = fig_ui.add_subplot(gs[2, :])

ax_text.axis("off")


def get_frame_data(idx):
    # 获取该 episode 在整个数据集中的绝对索引
    abs_idx = int(episode_data[idx]["index"])
    return dataset[abs_idx]


# 初始化渲染
init_f = get_frame_data(0)
img_a = ax_agent.imshow(init_f["observation.image"].permute(1, 2, 0).numpy())
img_w = ax_wrist.imshow(init_f["observation.wrist_image"].permute(1, 2, 0).numpy())

ax_agent.set_title("Agent View")
ax_agent.axis("off")
ax_wrist.set_title("Wrist View")
ax_wrist.axis("off")

text_display = ax_text.text(
    0.5,
    0.5,
    "",
    ha="center",
    va="center",
    fontsize=11,
    family="monospace",
    bbox=dict(facecolor="white", alpha=0.5),
)

frame_slider = Slider(
    ax=ax_slider,
    label="Frame ",
    valmin=0,
    valmax=num_frames_in_ep - 1,
    valinit=0,
    valstep=1,
)


def update(val):
    idx = int(frame_slider.val)
    f_data = get_frame_data(idx)

    img_a.set_data(f_data["observation.image"].permute(1, 2, 0).numpy())
    img_w.set_data(f_data["observation.wrist_image"].permute(1, 2, 0).numpy())

    s_v = f_data["observation.state"].numpy()
    a_v = f_data["actions"].numpy()
    text_display.set_text(
        f"FRAME: {idx:04d} | EPISODE: {EPISODE_ID}\n"
        f"STATE : {np.round(s_v, 4)}\n"
        f"ACTION: {np.round(a_v, 4)}"
    )
    fig_ui.canvas.draw_idle()


frame_slider.on_changed(update)
update(0)

print("Opening visualizer...")
plt.show()
