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
from pathlib import Path

import wandb
from lerobot.rl.wandb_utils import get_safe_wandb_artifact_name
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

os.environ["DISPLAY"] = ":11.0"

MyPolicy.set_visible_cuda_devices("1")
device = torch.device("cuda:0")

# Micro-batch per forward; effective batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS.
PER_DEVICE_BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 2

DATASET_REPO = "auboI10"
# DATASET_NAME = "data_w_shadow_x264"
# DATASET_NAME = "data_w_shadow_h264_znear0001"
DATASET_NAME = "aloha_sim_transfer_cube_human"
# DATASET_ROOT = f"/home/ningyu/MyI10Tele/{DATASET_NAME}/"
DATASET_ROOT = os.path.expanduser(
    f"~/openpi-cache/huggingface/lerobot/lerobot/{DATASET_NAME}"
)

dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
input_features, output_features = MyPolicy.input_output_features_from_metadata(
    dataset_metadata
)

# --- 初始化 ACT 模型 ---
is_finetuning = False
# pretrained_model_id = "lerobot/act_aloha_sim_insertion_human"
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
    batch_size=PER_DEVICE_BATCH_SIZE,
    shuffle=True,
    pin_memory=device.type != "cpu",
    persistent_workers=True,
    drop_last=True,
)
_effective_bs = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
print(
    f"Dataset size: {len(dataset)}, micro-batches/epoch: {len(dataloader)}, "
    f"micro_batch={PER_DEVICE_BATCH_SIZE}, grad_accum={GRADIENT_ACCUMULATION_STEPS} "
    f"(effective batch ≈ {_effective_bs})"
)


# --- 训练循环 ---

best_loss = float("inf")
# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 100
log_freq = 100
# save_dir = '.ckpt/auboI10_act_w_2_view_temporal_ensemble_coeff09'
save_dir = f".ckpt/{pretrained_model_id.split('/')[-1]}/{DATASET_NAME}"

# --- WandB init (optional via env) ---
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "aubo-i10-fintune")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "phil_ning")
_global_step = 0
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    dir=save_dir,
    config={
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accum": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch": _effective_bs,
        "training_steps": training_steps,
    },
    resume="allow",
)
print(f"WandB initialized: {wandb.run.get_url()}")
# tqdm: one unit per optimizer.step() (after grad accum), not per micro-batch.
_dl_len = len(dataloader)
_steps_per_epoch = _dl_len // GRADIENT_ACCUMULATION_STEPS + (
    1 if _dl_len % GRADIENT_ACCUMULATION_STEPS else 0
)
_total_optimizer_steps = training_steps * _steps_per_epoch
pbar = tqdm(total=_total_optimizer_steps, desc="Train", dynamic_ncols=True, leave=True)

for epoch in range(training_steps):
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    accum_step = 0

    for batch in dataloader:
        batch = MyPolicy.move_batch_to_device(batch, device)

        loss, _ = policy(batch)
        scaled = loss / GRADIENT_ACCUMULATION_STEPS
        scaled.backward()

        accum_step += 1
        epoch_loss += loss.item()
        pbar.set_postfix(
            epoch=f"{epoch + 1}/{training_steps}", Loss=f"{loss.item():.4f}"
        )

        if accum_step >= GRADIENT_ACCUMULATION_STEPS:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            _global_step += 1
            wandb.log(
                {"train/loss": loss.item(), "train/step": _global_step},
                step=_global_step,
            )
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0
            pbar.update(1)

    if accum_step > 0:
        _partial_scale = GRADIENT_ACCUMULATION_STEPS / accum_step
        for p in policy.parameters():
            if p.grad is not None:
                p.grad.mul_(_partial_scale)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        _global_step += 1
        wandb.log(
            {"train/loss": loss.item(), "train/step": _global_step},
            step=_global_step,
        )
        optimizer.zero_grad(set_to_none=True)
        pbar.update(1)

    avg_loss = epoch_loss / len(dataloader)
    if (epoch + 1) % log_freq == 0 or epoch == training_steps - 1:
        print(f"Epoch {epoch + 1}/{training_steps} 完成. 平均 Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        policy.save_pretrained(save_dir)
        print(f"已保存新的最优模型至 {save_dir} (Loss: {best_loss:.4f})")


pbar.close()
print("训练结束！")
