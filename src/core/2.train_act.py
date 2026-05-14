#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # ACT 策略训练与微调 (基于 LeRobot)
# 本 Notebook 包含使用 LeRobot 库进行 ACT 训练的完整流程。它将分步导入依赖、加载数据集、配置并初始化模型，最后执行训练与模型保存。


import torch
from tqdm.auto import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.datasets.factory import resolve_delta_timestamps

from core.my_policy import MyPolicy
from dist.dist import AddGaussianNoise

from torchvision import transforms

import os

os.environ["DISPLAY"] = ":11.0"

MyPolicy.set_visible_cuda_devices("1")
device = torch.device("cuda:0")

DATASET_REPO = "auboI10"
DATASET_NAME = "data_w_shadow_x264"
DATASET_ROOT = f"/home/ningyu/MyI10Tele/{DATASET_NAME}/"

dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
input_features, output_features = MyPolicy.input_output_features_from_metadata(
    dataset_metadata
)

# --- 初始化 ACT 模型 ---
is_finetuning = False
pretrained_model_id = "lerobot/act_aloha_sim_insertion_human"
pretrained_model_id = "lerobot/act_aloha_sim_transfer_cube_human"
if is_finetuning:
    print(f"加载预训练模型用于微调: {pretrained_model_id}")
    policy = ACTPolicy.from_pretrained(pretrained_model_id)

    # 冻结视觉 Backbone 以加速微调
    for param in policy.model.backbone.parameters():
        param.requires_grad = False
    print("视觉主干网络已冻结。")
    cfg = policy.config
else:
    print("从头初始化 ACT 模型...")

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    # Keep all dataset inputs (including observation.wrist_image) for multi-view ACT.

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=100,
        n_action_steps=1,
        temporal_ensemble_coeff=0.9,
        dropout=0.1,
        device=str(device),
    )
    # This allows us to construct the data with action chunking
    # We can now instantiate our policy with this config and the dataset stats.

    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)

delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
policy.train()
policy.to(device)


# note Create a transformation pipeline that converts a PIL image to a tensor, then adds noise.
transform = transforms.Compose(
    [
        AddGaussianNoise(mean=0.0, std=0.01),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
    ]
)


# We can then instantiate the dataset with these delta_timestamps configuration.
dataset = LeRobotDataset(
    DATASET_REPO,
    delta_timestamps=delta_timestamps,
    root=DATASET_ROOT,
    image_transforms=transform,
    video_backend="torchcodec",
)

# Then we create our optimizer and dataloader for offline training.
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=256,
    shuffle=True,
    pin_memory=device.type != "cpu",
    persistent_workers=True,
    drop_last=True,
)


# --- 训练循环 ---

best_loss = float("inf")
# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 100
log_freq = 100
# save_dir = '.ckpt/auboI10_act_w_2_view_temporal_ensemble_coeff09'
save_dir = f".ckpt/{DATASET_NAME}"

# One bar for the whole run (updates in place). Avoids Jupyter stacking many tqdm widgets.
_total_batches = training_steps * len(dataloader)
pbar = tqdm(total=_total_batches, desc="Train", dynamic_ncols=True, leave=True)

for epoch in range(training_steps):
    epoch_loss = 0.0

    for batch in dataloader:
        batch = MyPolicy.move_batch_to_device(batch, device)

        optimizer.zero_grad()
        loss, _ = policy(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix(
            epoch=f"{epoch + 1}/{training_steps}", Loss=f"{loss.item():.4f}"
        )

    avg_loss = epoch_loss / len(dataloader)
    if (epoch + 1) % log_freq == 0 or epoch == training_steps - 1:
        print(f"Epoch {epoch + 1}/{training_steps} 完成. 平均 Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        policy.save_pretrained(save_dir)
        print(f"已保存新的最优模型至 {save_dir} (Loss: {best_loss:.4f})")
    if avg_loss < 0.03:
        print(f"训练结束！")
        pbar.close()
        break


pbar.close()
print("训练结束！")
