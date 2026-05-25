#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PI0 fine-tuning aligned with LeRobot docs + Aloha sim recipe (Issue #1951).

References:
  - https://huggingface.co/docs/lerobot/en/pi0
  - https://github.com/huggingface/lerobot/issues/1951 (aloha_sim_transfer_cube_human, 100k steps)
  - Base weights: lerobot/pi0_base (local via resolve_pi0_pretrained_path)
"""

import os
from contextlib import nullcontext
from pathlib import Path

_hf_root = Path.home() / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_hf_root))
os.environ.setdefault("HF_HUB_CACHE", str(_hf_root / "hub"))
_allow_hub = False
_allow_hub_tokenizer = False

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("DISPLAY", ":11.0")

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

from core.my_policy import (
    MyPIPolicy,
    MyPolicy,
    resolve_paligemma_tokenizer_path,
    resolve_pi0_pretrained_path,
)

# --- Previous imports ---
# from dist.dist import AddGaussianNoise
# from torchvision import transforms
# import wandb
# from lerobot.rl.wandb_utils import get_safe_wandb_artifact_name
# from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

MyPolicy.set_visible_cuda_devices("0")
device = torch.device("cuda")

# --- Previous (custom) settings ---
# PI0_NUM_EPOCHS = 250
# PER_DEVICE_BATCH_SIZE = 16
# GRADIENT_ACCUMULATION_STEPS = 4
# COMPILE_PI0_MODEL = True
# USE_GRADIENT_CHECKPOINTING = not COMPILE_PI0_MODEL
# PI0_COMPILE_MODE = "max-autotune-no-cudagraphs"
# DATASET_REPO = "auboI10"
# DATASET_NAME = "data_w_shadow_h264_znear0001"
# train_expert_only=True, chunk_size=100, n_action_steps=1
# transform = AddGaussianNoise(...)
# best_loss epoch save loop

# --- Official-aligned + VRAM profiles (31GB GPU: do NOT use fast without 40GB+) ---
#   safe   — ckpt on, batch 15, no compile (you previously ~87% VRAM, stable)
#   memory — ckpt on, micro_batch 6 × accum 4, autocast (lower peak VRAM)
#   expert — ckpt on, train_expert_only (less VRAM, faster; slightly different recipe)
#   fast   — compile + no ckpt: needs ~40GB+; OOM on 32GB even at batch 8
# Override: PI0_TRAIN_PROFILE=memory python 5.train_pi.py
PI0_TRAIN_PROFILE = os.environ.get("PI0_TRAIN_PROFILE", "safe")

TRAINING_STEPS = 100_000
LOG_FREQ = 200
SAVE_FREQ = 25_000

if PI0_TRAIN_PROFILE == "fast":
    COMPILE_PI0_MODEL = True
    USE_GRADIENT_CHECKPOINTING = False
    PI0_COMPILE_MODE = "max-autotune-no-cudagraphs"
    PER_DEVICE_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 1
    USE_AMP = False
    TRAIN_EXPERT_ONLY = False
elif PI0_TRAIN_PROFILE == "memory":
    COMPILE_PI0_MODEL = False
    USE_GRADIENT_CHECKPOINTING = True
    PI0_COMPILE_MODE = "default"
    PER_DEVICE_BATCH_SIZE = 6
    GRADIENT_ACCUMULATION_STEPS = 4
    USE_AMP = True
    TRAIN_EXPERT_ONLY = False
elif PI0_TRAIN_PROFILE == "expert":
    COMPILE_PI0_MODEL = False
    USE_GRADIENT_CHECKPOINTING = True
    PI0_COMPILE_MODE = "default"
    PER_DEVICE_BATCH_SIZE = 10
    GRADIENT_ACCUMULATION_STEPS = 2
    USE_AMP = True
    TRAIN_EXPERT_ONLY = True
else:  # safe
    COMPILE_PI0_MODEL = False
    USE_GRADIENT_CHECKPOINTING = True
    PI0_COMPILE_MODE = "default"
    PER_DEVICE_BATCH_SIZE = 15
    GRADIENT_ACCUMULATION_STEPS = 1
    USE_AMP = False
    TRAIN_EXPERT_ONLY = False

DATASET_REPO = "lerobot/aloha_sim_transfer_cube_human"
DATASET_NAME = "aloha_sim_transfer_cube_human"
DATASET_ROOT = os.path.expanduser(
    f"~/openpi-cache/huggingface/lerobot/lerobot/{DATASET_NAME}"
)

PI0_PRETRAINED_DIR: str | None = None
pretrained_model_id, pi_pretrained_local_only = resolve_pi0_pretrained_path(
    PI0_PRETRAINED_DIR,
    allow_hub_download=_allow_hub,
)
SAVE_DIR = f".ckpt/{pretrained_model_id.split('/')[-1]}/{DATASET_NAME}"

dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
input_features, output_features = MyPolicy.input_output_features_from_metadata(
    dataset_metadata
)

print(
    f"Loading PI0 base: {pretrained_model_id} (local_files_only={pi_pretrained_local_only})"
)
cfg = PI0Config(
    input_features=input_features,
    output_features=output_features,
    # LeRobot PI0 defaults; Aloha sim often uses empty_cameras=2 for 3-view layout (#1951).
    chunk_size=50,
    n_action_steps=50,
    empty_cameras=2,
    image_resolution=(224, 224),
    dtype="bfloat16",
    use_amp=USE_AMP,
    compile_model=COMPILE_PI0_MODEL,
    compile_mode=PI0_COMPILE_MODE if COMPILE_PI0_MODEL else "default",
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    freeze_vision_encoder=True,
    train_expert_only=TRAIN_EXPERT_ONLY,
    device=str(device),
)
_effective_bs = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
print(
    f"PI0 train profile={PI0_TRAIN_PROFILE!r}  "
    f"micro_batch={PER_DEVICE_BATCH_SIZE}  grad_accum={GRADIENT_ACCUMULATION_STEPS}  "
    f"effective_batch={_effective_bs}  compile={cfg.compile_model}  "
    f"grad_ckpt={cfg.gradient_checkpointing}  use_amp={cfg.use_amp}  "
    f"train_expert_only={cfg.train_expert_only}"
)
print(
    "PI0 config: "
    f"chunk={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}, "
    f"empty_cameras={cfg.empty_cameras}, freeze_vision={cfg.freeze_vision_encoder}, "
    f"train_expert_only={cfg.train_expert_only}"
)
if PI0_TRAIN_PROFILE == "fast":
    print(
        "WARNING: profile=fast needs ~40GB+ VRAM (compile + no checkpointing). "
        "On 32GB use safe / memory / expert instead."
    )
if COMPILE_PI0_MODEL:
    print(
        "torch.compile warmup: first ~20–50 steps are slow; steady-state is much faster."
    )
if USE_AMP:
    print(
        "use_amp=True: training forward runs under autocast(bfloat16). "
        "Weights are already bf16; main VRAM saver is gradient_checkpointing + smaller micro-batch."
    )
policy = PI0Policy.from_pretrained(
    pretrained_model_id,
    config=cfg,
    dataset_stats=dataset_metadata.stats,
    local_files_only=pi_pretrained_local_only,
)

delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
policy.train()
policy.to(device)
print(
    f"Model on {device}, trainable params: "
    f"{sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6:.1f}M"
)

_tokenizer_src, _tokenizer_local = resolve_paligemma_tokenizer_path(
    allow_hub_download=_allow_hub_tokenizer,
)
print(f"Tokenizer: {_tokenizer_src} (local_files_only={_tokenizer_local})")
tokenizer = AutoTokenizer.from_pretrained(
    _tokenizer_src, local_files_only=_tokenizer_local
)
tokenizer.padding_side = "right"
pi_lang = MyPIPolicy(tokenizer, cfg.tokenizer_max_length)

dataset = LeRobotDataset(
    DATASET_REPO,
    delta_timestamps=delta_timestamps,
    root=DATASET_ROOT,
    image_transforms=None,
    # image_transforms=transform,  # previous AddGaussianNoise pipeline
    video_backend="torchcodec",
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=PER_DEVICE_BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
print(
    f"Dataset size: {len(dataset)}, micro-batches/epoch: {len(dataloader)}, "
    f"target optimizer steps={TRAINING_STEPS}"
)

optimizer_cfg = cfg.get_optimizer_preset()
optimizer = optimizer_cfg.build(filter(lambda p: p.requires_grad, policy.parameters()))
scheduler = cfg.get_scheduler_preset().build(
    optimizer, num_training_steps=TRAINING_STEPS
)

_amp_ctx = (
    torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if USE_AMP and device.type == "cuda"
    else nullcontext()
)

pbar = tqdm(total=TRAINING_STEPS, desc="Train PI0", dynamic_ncols=True, leave=True)
_global_step = 0
accum_step = 0
optimizer.zero_grad(set_to_none=True)

while _global_step < TRAINING_STEPS:
    for batch in dataloader:
        batch = MyPolicy.move_batch_to_device(batch, device)
        batch = pi_lang.inject_language_tokens(batch, device)

        if COMPILE_PI0_MODEL:
            torch.compiler.cudagraph_mark_step_begin()

        with _amp_ctx:
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
        scheduler.step()
        _global_step += 1
        accum_step = 0
        optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix(
            step=f"{_global_step}/{TRAINING_STEPS}",
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )
        pbar.update(1)

        if _global_step % SAVE_FREQ == 0 or _global_step == TRAINING_STEPS:
            ckpt_dir = Path(SAVE_DIR) / f"checkpoint_{_global_step:06d}"
            policy.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint to {ckpt_dir}")

        if _global_step % LOG_FREQ == 0:
            print(
                f"Step {_global_step}/{TRAINING_STEPS}  loss={loss.item():.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if _global_step >= TRAINING_STEPS:
            break

pbar.close()
policy.save_pretrained(SAVE_DIR)
print(f"Training complete. Final weights: {SAVE_DIR}")

# --- Previous training loop (epoch-based, best-loss save, compile + grad accum) ---
# wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, dir=save_dir, ...)
# for epoch in range(PI0_NUM_EPOCHS):
#     for batch in dataloader:
#         batch = pi_lang.inject_language_tokens(batch, device)
#         if COMPILE_PI0_MODEL:
#             torch.compiler.cudagraph_mark_step_begin()
#         loss, loss_dict = policy(batch)
#         ...
#     if avg_loss < best_loss:
#         policy.save_pretrained(save_dir)
