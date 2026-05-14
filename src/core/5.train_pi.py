#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PI0 (π₀) Policy Fine-tuning via LeRobot
#
# PI0 = PaliGemma VLM + Gemma Action Expert, trained with Flow Matching.
#
# Prerequisites:
#   pip install "transformers @ git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0"
#   pip install sentencepiece
#   huggingface-cli login  (PaliGemma tokenizer is gated)
#
# Offline / local assets (default: no Hub download):
#   - PI0 weights: resolve_pi0_pretrained_path scans PI0_PRETRAINED, pretrained/pi0_*,
#     then HF hub cache models--lerobot--pi0*/snapshots (newest with model.safetensors).
#   - Tokenizer: resolve_paligemma_tokenizer_path scans PI0_TOKENIZER, pretrained/paligemma-3b-pt-224,
#     then HF hub cache for google/paligemma-3b-pt-224.
#   - Set PI0_ALLOW_HUB_DOWNLOAD=1 to allow Hub fallback for PI0 weights only.
#   - Tokenizer Hub fallback is separate: PI0_TOKENIZER_ALLOW_HUB_DOWNLOAD=1 (default off so offline runs
#     do not hit huggingface.co when PI0_ALLOW_HUB_DOWNLOAD is set for weights).
#
# Hugging Face cache defaults (set before importing torch/transformers; override with env):
#   HF_HOME=~/.cache/huggingface, HF_HUB_CACHE=~/.cache/huggingface/hub
#
# train_expert_only: only VLM weights are frozen (requires_grad=False); forward still runs full
# PaliGemma (vision + language + prefix transformer). Gradients still flow through VLM activations
# into the expert, so most FLOPs remain — you save optimizer state / param updates, not the big forward.

import os
from pathlib import Path


_hf_root = Path.home() / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_hf_root))
os.environ.setdefault("HF_HUB_CACHE", str(_hf_root / "hub"))
_allow_hub = os.environ.get("PI0_ALLOW_HUB_DOWNLOAD", "").lower() in (
    "1",
    "true",
    "yes",
)
_allow_hub_tokenizer = os.environ.get(
    "PI0_TOKENIZER_ALLOW_HUB_DOWNLOAD", ""
).lower() in ("1", "true", "yes")

# Reduce allocator fragmentation (helps when compile / cudagraphs reserve pools).
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PI0_NUM_EPOCHS = int(os.environ.get("PI0_NUM_EPOCHS", "25"))
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.datasets.factory import resolve_delta_timestamps
from dist.dist import AddGaussianNoise
from torchvision import transforms

from core.my_policy import (
    MyPIPolicy,
    MyPolicy,
    resolve_paligemma_tokenizer_path,
    resolve_pi0_pretrained_path,
)

MyPolicy.set_visible_cuda_devices("0")
device = torch.device("cuda:0")

# torch.compile + activation checkpointing: AOT partitioner's functionalize_rng_ops walks joint-graph
# nodes marked MUST_RECOMPUTE that carry RNG (nondeterministic_seeded), e.g. flash SDPA internals.
# After min-cut partition, those nodes may not appear by the same name in the bw subgraph → KeyError.
# Fix at source: disable checkpointing when compiling so has_recomputable_rng_ops is false (see PyTorch
# functorch partitioners.py / issues around checkpoint + compile + RNG).
COMPILE_PI0_MODEL = True
USE_GRADIENT_CHECKPOINTING = not COMPILE_PI0_MODEL

# torch.compile: "max-autotune" enables CUDA Graphs (faster steady-state, risk of overwrite errors).
# Default is max-autotune-no-cudagraphs (stable). Enable graphs: PI0_CUDAGRAPHS=1 or PI0_COMPILE_MODE=max-autotune
# (each micro-batch calls torch.compiler.cudagraph_mark_step_begin() to play nice with grad accumulation).
_default_compile = "max-autotune-no-cudagraphs"
if os.environ.get("PI0_CUDAGRAPHS", "").lower() in ("1", "true", "yes"):
    _default_compile = "max-autotune"
PI0_COMPILE_MODE = os.environ.get("PI0_COMPILE_MODE", _default_compile)

# VRAM: micro-batch per forward/backward; gradient accumulation matches previous 64 effective batch.
# Tune PI0_BATCH_SIZE / PI0_GRAD_ACCUM if OOM persists (e.g. 4 and 16).
PER_DEVICE_BATCH_SIZE = int(os.environ.get("PI0_BATCH_SIZE", "16"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("PI0_GRAD_ACCUM", "4"))

DATASET_REPO = "auboI10"
# DATASET_ROOT = "/home/ningyu/MyI10Tele/data2/"
DATASET_NAME = "data_w_shadow_x264"
DATASET_ROOT = f"/home/ningyu/MyI10Tele/{DATASET_NAME}/"

dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
input_features, output_features = MyPolicy.input_output_features_from_metadata(
    dataset_metadata
)

# --- Initialize PI0 model ---
is_finetuning = True
# Optional: folder with model.safetensors; otherwise env PI0_PRETRAINED / hub cache / pretrained/*.
PI0_PRETRAINED_DIR: str | None = None

pretrained_model_id, pi_pretrained_local_only = resolve_pi0_pretrained_path(
    PI0_PRETRAINED_DIR,
    allow_hub_download=_allow_hub,
)

if is_finetuning:
    print(
        f"Loading pretrained model: {pretrained_model_id} (local_files_only={pi_pretrained_local_only})"
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
        device=str(device),
    )
    policy = PI0Policy.from_pretrained(
        pretrained_model_id,
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
        chunk_size=50,
        n_action_steps=50,
        dtype="bfloat16",
        gradient_checkpointing=True,
        device=str(device),
    )
    policy = PI0Policy(cfg, dataset_stats=dataset_metadata.stats)

delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
policy.train()
policy.to(device)
print(
    f"Model on {device}, param count: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M"
)
transform = transforms.Compose(
    [
        AddGaussianNoise(mean=0.0, std=0.01),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
    ]
)


# --- Tokenizer for language conditioning ---
# PI0 forward() expects batch["observation.language_tokens"] and
# batch["observation.language_attention_mask"], produced by PaliGemma tokenizer.
_tokenizer_src, _tokenizer_local = resolve_paligemma_tokenizer_path(
    allow_hub_download=_allow_hub_tokenizer,
)
print(f"Tokenizer from: {_tokenizer_src} (local_files_only={_tokenizer_local})")
tokenizer = AutoTokenizer.from_pretrained(
    _tokenizer_src, local_files_only=_tokenizer_local
)
tokenizer.padding_side = "right"
TOKENIZER_MAX_LENGTH = cfg.tokenizer_max_length  # 48
pi_lang = MyPIPolicy(tokenizer, TOKENIZER_MAX_LENGTH)

# --- Dataset & DataLoader ---
dataset = LeRobotDataset(
    DATASET_REPO,
    delta_timestamps=delta_timestamps,
    root=DATASET_ROOT,
    image_transforms=transform,
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
_effective_bs = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
print(
    f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}, "
    f"micro_batch={PER_DEVICE_BATCH_SIZE}, grad_accum={GRADIENT_ACCUMULATION_STEPS} "
    f"(effective batch ≈ {_effective_bs})"
)

# --- Optimizer (follow openpi AdamW defaults) ---
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, policy.parameters()),
    lr=cfg.optimizer_lr,
    betas=cfg.optimizer_betas,
    eps=cfg.optimizer_eps,
    weight_decay=cfg.optimizer_weight_decay,
)

# --- Training loop ---
# Outer loop: full passes over the dataset (epochs).
# tqdm matches ACT-style scripts: one unit per optimizer.step() (not per micro-batch),
# so totals and it/s are comparable to DataLoader batch_size runs without grad accum.
best_loss = float("inf")
log_freq = 50
save_dir = f".ckpt/{DATASET_NAME}"

_dl_len = len(dataloader)
_steps_per_epoch = _dl_len // GRADIENT_ACCUMULATION_STEPS + (
    1 if _dl_len % GRADIENT_ACCUMULATION_STEPS else 0
)
_total_micro_batches = PI0_NUM_EPOCHS * _dl_len
_total_optimizer_steps = PI0_NUM_EPOCHS * _steps_per_epoch
pbar = tqdm(
    total=_total_optimizer_steps, desc="Train PI0", dynamic_ncols=True, leave=True
)
print(
    f"Training {PI0_NUM_EPOCHS} epoch(s); {_dl_len} micro-batches/epoch, "
    f"grad_accum={GRADIENT_ACCUMULATION_STEPS} → {_steps_per_epoch} optimizer steps/epoch "
    f"({_total_optimizer_steps} total, tqdm). {_total_micro_batches} forwards total. "
    f"Set PI0_NUM_EPOCHS to reduce wall time."
)

for epoch in range(PI0_NUM_EPOCHS):
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    accum_step = 0

    for batch in dataloader:
        batch = MyPolicy.move_batch_to_device(batch, device)
        batch = pi_lang.inject_language_tokens(batch, device)

        if COMPILE_PI0_MODEL:
            # Separates CUDA graph steps when Inductor cudagraphs are on (e.g. PI0_COMPILE_MODE=max-autotune).
            torch.compiler.cudagraph_mark_step_begin()

        loss, loss_dict = policy(batch)
        scaled = loss / GRADIENT_ACCUMULATION_STEPS
        scaled.backward()

        accum_step += 1
        epoch_loss += loss.item()
        pbar.set_postfix(
            epoch=f"{epoch + 1}/{PI0_NUM_EPOCHS}", Loss=f"{loss.item():.4f}"
        )

        if accum_step >= GRADIENT_ACCUMULATION_STEPS:
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=cfg.optimizer_grad_clip_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0
            pbar.update(1)

    if accum_step > 0:
        # Partial last group: backward used loss/GRAD_ACCUM each time; rescale so mean matches m micro-batches.
        _partial_scale = GRADIENT_ACCUMULATION_STEPS / accum_step
        for p in policy.parameters():
            if p.grad is not None:
                p.grad.mul_(_partial_scale)
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=cfg.optimizer_grad_clip_norm
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pbar.update(1)

    avg_loss = epoch_loss / len(dataloader)
    if (epoch + 1) % log_freq == 0 or epoch == PI0_NUM_EPOCHS - 1:
        print(f"Epoch {epoch + 1}/{PI0_NUM_EPOCHS}, avg_loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        policy.save_pretrained(save_dir)
        print(f"Saved best model to {save_dir} (Loss: {best_loss:.4f})")

pbar.close()
print("Training complete!")
