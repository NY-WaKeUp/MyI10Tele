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

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.datasets.factory import resolve_delta_timestamps

from core.my_policy import MyPIPolicy, MyPolicy

MyPolicy.set_visible_cuda_devices("0")
device = torch.device("cuda:0")

DATASET_REPO = "auboI10"
DATASET_ROOT = "/home/ningyu/MyI10Tele/data2/"

dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
input_features, output_features = MyPolicy.input_output_features_from_metadata(dataset_metadata)

# --- Initialize PI0 model ---
is_finetuning = True
pretrained_model_id = "lerobot/pi0_base"

if is_finetuning:
    print(f"Loading pretrained model: {pretrained_model_id}")
    cfg = PI0Config(
        input_features=input_features,
        output_features=output_features,
        compile_model=True,
        dtype="bfloat16",
        gradient_checkpointing=True,
        train_expert_only=True,
    )
    policy = PI0Policy.from_pretrained(
        pretrained_model_id,
        config=cfg,
        dataset_stats=dataset_metadata.stats,
    )
    print("Train expert only for fine-tuning.")
else:
    print("Initializing PI0 model from scratch...")
    cfg = PI0Config(
        input_features=input_features,
        output_features=output_features,
        chunk_size=50,
        n_action_steps=50,
        dtype="bfloat16",
        gradient_checkpointing=True,
    )
    policy = PI0Policy(cfg, dataset_stats=dataset_metadata.stats)

delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
policy.train()
policy.to(device)
print(f"Model on {device}, param count: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")

# --- Tokenizer for language conditioning ---
# PI0 forward() expects batch["observation.language_tokens"] and
# batch["observation.language_attention_mask"], produced by PaliGemma tokenizer.
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
tokenizer.padding_side = "right"
TOKENIZER_MAX_LENGTH = cfg.tokenizer_max_length  # 48
pi_lang = MyPIPolicy(tokenizer, TOKENIZER_MAX_LENGTH)

# --- Dataset & DataLoader ---
dataset = LeRobotDataset(
    DATASET_REPO,
    delta_timestamps=delta_timestamps,
    root=DATASET_ROOT,
    video_backend="torchcodec",
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
print(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")

# --- Optimizer (follow openpi AdamW defaults) ---
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, policy.parameters()),
    lr=cfg.optimizer_lr,
    betas=cfg.optimizer_betas,
    eps=cfg.optimizer_eps,
    weight_decay=cfg.optimizer_weight_decay,
)

# --- Training loop ---
best_loss = float("inf")
training_steps = 3000
log_freq = 50
save_dir = ".ckpt/auboI10_pi0_finetuned"

_total_batches = training_steps * len(dataloader)
pbar = tqdm(total=_total_batches, desc="Train PI0", dynamic_ncols=True, leave=True)

for epoch in range(training_steps):
    epoch_loss = 0.0

    for batch in dataloader:
        batch = MyPolicy.move_batch_to_device(batch, device)
        batch = pi_lang.inject_language_tokens(batch, device)

        optimizer.zero_grad()
        loss, loss_dict = policy(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.optimizer_grad_clip_norm)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix(epoch=f"{epoch + 1}/{training_steps}", Loss=f"{loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    if (epoch + 1) % log_freq == 0 or epoch == training_steps - 1:
        print(f"Epoch {epoch + 1}/{training_steps}, avg_loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        policy.save_pretrained(save_dir)
        print(f"Saved best model to {save_dir} (Loss: {best_loss:.4f})")

pbar.close()
print("Training complete!")
