#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACT training aligned with LeRobot official Aloha transfer-cube run.

Official references:
  - Model card: https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human
  - WandB run: https://wandb.ai/aliberts/lerobot/runs/720l37xb (80k optimizer steps, batch 8)
"""

import os
from pathlib import Path

import torch
import wandb
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from tqdm.auto import tqdm

from core.my_policy import MyPolicy

# --- Previous imports (custom pipeline) ---
# from dist.dist import AddGaussianNoise
# from torchvision import transforms
# from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

os.environ["DISPLAY"] = ":11.0"

MyPolicy.set_visible_cuda_devices("1")
device = torch.device("cuda:0")

# --- Previous (custom) hyperparameters ---
# PER_DEVICE_BATCH_SIZE = 128
# GRADIENT_ACCUMULATION_STEPS = 2
# training_steps = 1000  # outer loop: full epochs over the dataset
# log_freq = 100
# DATASET_REPO = "auboI10"
# DATASET_NAME = "data_w_shadow_x264"
# DATASET_NAME = "data_w_shadow_h264_znear0001"
# DATASET_ROOT = f"/home/ningyu/MyI10Tele/{DATASET_NAME}/"
# save_dir = '.ckpt/auboI10_act_w_2_view_temporal_ensemble_coeff09'

# --- Official-ish training schedule (WandB 720l37xb: 80k steps, ~640k samples -> batch 8) ---
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
TRAINING_STEPS = 80_000
LOG_FREQ = 1000
SAVE_FREQ = 10_000

# Local mirror of https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human
DATASET_REPO = "lerobot/aloha_sim_transfer_cube_human"
DATASET_NAME = "aloha_sim_transfer_cube_human"
DATASET_ROOT = os.path.expanduser(
    f"~/openpi-cache/huggingface/lerobot/lerobot/{DATASET_NAME}"
)

# Hub id used for naming; training starts from scratch like the official run.
PRETRAINED_MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"
SAVE_DIR = f".ckpt/{PRETRAINED_MODEL_ID.split('/')[-1]}/{DATASET_NAME}"

dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
input_features, output_features = MyPolicy.input_output_features_from_metadata(
    dataset_metadata
)

# --- Previous model init (scratch / finetune branches) ---
# is_finetuning = False
# pretrained_model_id = "lerobot/act_aloha_sim_transfer_cube_human"
# if is_finetuning:
#     print(f"加载预训练模型用于微调: {pretrained_model_id}")
#     policy = ACTPolicy.from_pretrained(pretrained_model_id)
#     for param in policy.model.backbone.parameters():
#         param.requires_grad = False
#     print("视觉主干网络已冻结。")
#     cfg = policy.config
# else:
#     print("从头初始化 ACT 模型...")
#     cfg = ACTConfig(
#         input_features=input_features,
#         output_features=output_features,
#         chunk_size=100,
#         n_action_steps=1,
#         temporal_ensemble_coeff=0.9,
#         dropout=0.1,
#         device=str(device),
#     )
#     policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)

# ACTConfig defaults match official WandB: chunk/n_action=100, temporal_ensemble=None, lr=1e-5.
cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
    chunk_size=100,
    n_action_steps=100,
    temporal_ensemble_coeff=None,
    dropout=0.1,
    device=str(device),
)
print(
    "ACT config (official defaults): "
    f"chunk_size={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}, "
    f"temporal_ensemble_coeff={cfg.temporal_ensemble_coeff}, lr={cfg.optimizer_lr}"
)
policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)

delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
policy.train()
policy.to(device)

# --- Previous image augmentation ---
# transform = transforms.Compose(
#     [
#         AddGaussianNoise(mean=0.0, std=0.01),
#         transforms.Lambda(lambda x: x.clamp(0, 1)),
#     ]
# )

dataset = LeRobotDataset(
    DATASET_REPO,
    delta_timestamps=delta_timestamps,
    root=DATASET_ROOT,
    image_transforms=None,
    # image_transforms=transform,
    video_backend="torchcodec",
)

optimizer_cfg = cfg.get_optimizer_preset()
optimizer = optimizer_cfg.build(policy.parameters())
# --- Previous optimizer ---
# optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
# optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)

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
    f"(effective batch = {_effective_bs}), target optimizer steps={TRAINING_STEPS}"
)

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "aubo-i10-fintune")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "phil_ning")
_global_step = 0
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    dir=SAVE_DIR,
    config={
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accum": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch": _effective_bs,
        "training_steps": TRAINING_STEPS,
        "dataset_repo": DATASET_REPO,
        "official_wandb_run": "https://wandb.ai/aliberts/lerobot/runs/720l37xb",
    },
    resume="allow",
)
print(f"WandB initialized: {wandb.run.get_url()}")

# --- Previous training loop (epoch-based, save best loss) ---
# best_loss = float("inf")
# _dl_len = len(dataloader)
# _steps_per_epoch = _dl_len // GRADIENT_ACCUMULATION_STEPS + (
#     1 if _dl_len % GRADIENT_ACCUMULATION_STEPS else 0
# )
# _total_optimizer_steps = training_steps * _steps_per_epoch
# pbar = tqdm(total=_total_optimizer_steps, desc="Train", dynamic_ncols=True, leave=True)
# for epoch in range(training_steps):
#     epoch_loss = 0.0
#     optimizer.zero_grad(set_to_none=True)
#     accum_step = 0
#     for batch in dataloader:
#         batch = MyPolicy.move_batch_to_device(batch, device)
#         loss, _ = policy(batch)
#         scaled = loss / GRADIENT_ACCUMULATION_STEPS
#         scaled.backward()
#         accum_step += 1
#         epoch_loss += loss.item()
#         pbar.set_postfix(epoch=f"{epoch + 1}/{training_steps}", Loss=f"{loss.item():.4f}")
#         if accum_step >= GRADIENT_ACCUMULATION_STEPS:
#             torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
#             optimizer.step()
#             _global_step += 1
#             wandb.log({"train/loss": loss.item(), "train/step": _global_step}, step=_global_step)
#             optimizer.zero_grad(set_to_none=True)
#             accum_step = 0
#             pbar.update(1)
#     if accum_step > 0:
#         _partial_scale = GRADIENT_ACCUMULATION_STEPS / accum_step
#         for p in policy.parameters():
#             if p.grad is not None:
#                 p.grad.mul_(_partial_scale)
#         torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
#         optimizer.step()
#         _global_step += 1
#         wandb.log({"train/loss": loss.item(), "train/step": _global_step}, step=_global_step)
#         optimizer.zero_grad(set_to_none=True)
#         pbar.update(1)
#     avg_loss = epoch_loss / len(dataloader)
#     if (epoch + 1) % log_freq == 0 or epoch == training_steps - 1:
#         print(f"Epoch {epoch + 1}/{training_steps} 完成. 平均 Loss: {avg_loss:.4f}")
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         policy.save_pretrained(save_dir)
#         print(f"已保存新的最优模型至 {save_dir} (Loss: {best_loss:.4f})")
# pbar.close()
# print("训练结束！")

pbar = tqdm(total=TRAINING_STEPS, desc="Train ACT", dynamic_ncols=True, leave=True)
accum_step = 0
optimizer.zero_grad(set_to_none=True)

while _global_step < TRAINING_STEPS:
    for batch in dataloader:
        batch = MyPolicy.move_batch_to_device(batch, device)
        loss, _ = policy(batch)
        scaled = loss / GRADIENT_ACCUMULATION_STEPS
        scaled.backward()
        accum_step += 1

        if accum_step < GRADIENT_ACCUMULATION_STEPS:
            continue

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=optimizer_cfg.grad_clip_norm
        )
        optimizer.step()
        _global_step += 1
        accum_step = 0
        optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix(
            step=f"{_global_step}/{TRAINING_STEPS}", loss=f"{loss.item():.4f}"
        )
        pbar.update(1)
        wandb.log(
            {"train/loss": loss.item(), "train/step": _global_step}, step=_global_step
        )

        if _global_step % SAVE_FREQ == 0 or _global_step == TRAINING_STEPS:
            ckpt_dir = Path(SAVE_DIR) / f"checkpoint_{_global_step:06d}"
            policy.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint to {ckpt_dir}")

        if _global_step % LOG_FREQ == 0:
            print(f"Step {_global_step}/{TRAINING_STEPS}  loss={loss.item():.4f}")

        if _global_step >= TRAINING_STEPS:
            break

pbar.close()
policy.save_pretrained(SAVE_DIR)
print(f"Training complete. Final weights: {SAVE_DIR}")
