#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PI0 fine-tuning aligned with openpi ``TrainConfig`` (LeRobot PyTorch port).

Default preset mirrors ``pi0_auboI10_low_mem_finetune_qpos_k10`` in
``openpi/src/openpi/training/config.py``. Override via ``OPENPI_TRAIN_CONFIG`` env.

LeRobot has no LoRA variants; ``train_expert_only=True`` is the closest low-mem match.

Resume (example for ``data_auboI10_qpos_v30_continuous``)::

    cd src/core
    export PI0_RESUME=1
    # optional explicit checkpoint (default: latest under SAVE_DIR)
    # export PI0_RESUME_CKPT=".ckpt/pi0_auboI10_low_mem_finetune_qpos/data_auboI10_qpos_v30_continuous/checkpoint_042000"
    CUDA_VISIBLE_DEVICES=1 python 5.train_pi.py
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python 5.train_pi.py` from src/core without PYTHONPATH=src.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_hf_root = Path.home() / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_hf_root))
os.environ.setdefault("HF_HUB_CACHE", str(_hf_root / "hub"))
_allow_hub = False
_allow_hub_tokenizer = False

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("DISPLAY", ":11.0")

import torch
import wandb
from tqdm.auto import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.utils.constants import ACTION

from core.dataset_config import (
    OPENPI_TRAIN_CONFIG_EE_POSE,
    OPENPI_TRAIN_CONFIG_EE_POSE_K10,
    OPENPI_TRAIN_CONFIG_QPOS,
    OPENPI_TRAIN_CONFIG_QPOS_K10,
    REPO_NAME,
    dataset_root,
    teleop_ee_pose_root,
    teleop_qpos_root,
)
from core.my_policy import (
    MyPIPolicy,
    MyPolicy,
    action_delta_timestamps_sec,
    load_openpi_norm_stats,
    load_paligemma_tokenizer,
    openpi_norm_stats_path,
    openpi_norm_stats_to_lerobot,
    resolve_pi0_pretrained_path,
)

TRAINING_STATE_FILE = "training_state.pt"


def _checkpoint_step(ckpt_dir: Path) -> int:
    prefix = "checkpoint_"
    if not ckpt_dir.name.startswith(prefix):
        raise ValueError(f"Not a step checkpoint directory: {ckpt_dir}")
    return int(ckpt_dir.name.removeprefix(prefix))


def find_step_checkpoints(save_dir: Path) -> list[Path]:
    if not save_dir.is_dir():
        return []
    ckpts = [
        p for p in save_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint_")
    ]
    return sorted(ckpts, key=_checkpoint_step)


def resolve_resume_checkpoint(save_dir: Path) -> Path | None:
    """Resume from PI0_RESUME_CKPT or latest checkpoint_* when PI0_RESUME=1."""
    explicit = os.environ.get("PI0_RESUME_CKPT")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent / path).resolve()
        else:
            path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"PI0_RESUME_CKPT not found: {path}")
        if path.name.startswith("checkpoint_"):
            return path
        ckpts = find_step_checkpoints(path)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint_* under PI0_RESUME_CKPT: {path}")
        return ckpts[-1]

    if os.environ.get("PI0_RESUME", "0") != "1":
        return None

    ckpts = find_step_checkpoints(save_dir)
    if not ckpts:
        return None
    return ckpts[-1]


def find_wandb_run_id(save_dir: Path) -> str | None:
    wandb_dir = save_dir / "wandb"
    if not wandb_dir.is_dir():
        return None
    runs = sorted(wandb_dir.glob("run-*"))
    if not runs:
        return None
    return runs[-1].name.split("-")[-1]


def save_training_state(
    path: Path,
    *,
    global_step: int,
    accum_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    torch.save(
        {
            "global_step": global_step,
            "accum_step": accum_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def load_training_state(
    path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["global_step"]), int(state["accum_step"])


def fast_forward_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler, steps: int
) -> None:
    for _ in range(steps):
        scheduler.step()


@dataclass(frozen=True)
class OpenPITrainPreset:
    """Hyperparameters mirrored from openpi ``TrainConfig`` entries."""

    chunk_size: int
    action_horizon: int
    action_delta_stride: int
    num_train_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    asset_id: str
    use_qpos_delta_actions: bool
    # Aubo: 2 real cameras + 1 padded slot (openpi ``AuboInputs`` right_wrist mask=False).
    empty_cameras: int = 1
    log_interval: int = 100
    save_interval: int = 5000
    num_workers: int = os.cpu_count()
    pin_memory: bool = True
    persistent_workers: bool = True


OPENPI_TRAIN_PRESETS: dict[str, OpenPITrainPreset] = {
    OPENPI_TRAIN_CONFIG_QPOS: OpenPITrainPreset(
        chunk_size=50,
        action_horizon=50,
        action_delta_stride=1,
        num_train_steps=50_000,
        batch_size=2,
        gradient_accumulation_steps=4,
        asset_id="auboI10_qpos",
        use_qpos_delta_actions=True,
    ),
    OPENPI_TRAIN_CONFIG_QPOS_K10: OpenPITrainPreset(
        chunk_size=50,
        action_horizon=50,
        action_delta_stride=10,
        num_train_steps=30_000,
        batch_size=8,
        gradient_accumulation_steps=2,
        asset_id="auboI10_qpos_k10",
        use_qpos_delta_actions=True,
    ),
    OPENPI_TRAIN_CONFIG_EE_POSE: OpenPITrainPreset(
        chunk_size=50,
        action_horizon=10,
        action_delta_stride=1,
        num_train_steps=20_000,
        batch_size=8,
        gradient_accumulation_steps=1,
        asset_id="auboI10_ee_pose",
        use_qpos_delta_actions=False,
    ),
    OPENPI_TRAIN_CONFIG_EE_POSE_K10: OpenPITrainPreset(
        chunk_size=50,
        action_horizon=10,
        action_delta_stride=10,
        num_train_steps=30_000,
        batch_size=8,
        gradient_accumulation_steps=2,
        asset_id="auboI10_ee_pose_k10",
        use_qpos_delta_actions=False,
    ),
}

OPENPI_TRAIN_CONFIG_NAME = OPENPI_TRAIN_CONFIG_QPOS
if OPENPI_TRAIN_CONFIG_NAME not in OPENPI_TRAIN_PRESETS:
    raise ValueError(
        f"Unknown OPENPI_TRAIN_CONFIG={OPENPI_TRAIN_CONFIG_NAME!r}. "
        f"Choose from: {sorted(OPENPI_TRAIN_PRESETS)}"
    )
PRESET = OPENPI_TRAIN_PRESETS[OPENPI_TRAIN_CONFIG_NAME]
GRAD_ACCUM_STEPS = (
    int(os.environ["PI0_GRAD_ACCUM_STEPS"])
    if "PI0_GRAD_ACCUM_STEPS" in os.environ
    else PRESET.gradient_accumulation_steps
)
if GRAD_ACCUM_STEPS < 1:
    raise ValueError(f"PI0_GRAD_ACCUM_STEPS must be >= 1, got {GRAD_ACCUM_STEPS}")

_QPOS_CONFIGS = {OPENPI_TRAIN_CONFIG_QPOS, OPENPI_TRAIN_CONFIG_QPOS_K10}
_EE_CONFIGS = {OPENPI_TRAIN_CONFIG_EE_POSE, OPENPI_TRAIN_CONFIG_EE_POSE_K10}
if OPENPI_TRAIN_CONFIG_NAME in _QPOS_CONFIGS:
    _default_dataset_root = teleop_qpos_root()
elif OPENPI_TRAIN_CONFIG_NAME in _EE_CONFIGS:
    _default_dataset_root = teleop_ee_pose_root()
else:
    _default_dataset_root = dataset_root()

MyPolicy.set_visible_cuda_devices(os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
device = torch.device("cuda")

DATASET_REPO = REPO_NAME
DATASET_ROOT = os.environ.get("LEROBOT_ROOT", _default_dataset_root)
DATASET_NAME = Path(DATASET_ROOT).expanduser().name

PI0_PRETRAINED_DIR: str | None = None
pretrained_model_id, pi_pretrained_local_only = resolve_pi0_pretrained_path(
    PI0_PRETRAINED_DIR,
    allow_hub_download=_allow_hub,
)
SAVE_DIR = f".ckpt/{OPENPI_TRAIN_CONFIG_NAME}/{DATASET_NAME}"
SAVE_DIR_PATH = Path(SAVE_DIR)
if not SAVE_DIR_PATH.is_absolute():
    SAVE_DIR_PATH = (Path(__file__).resolve().parent / SAVE_DIR).resolve()

resume_ckpt_dir = resolve_resume_checkpoint(SAVE_DIR_PATH)

norm_stats_path = openpi_norm_stats_path(OPENPI_TRAIN_CONFIG_NAME, PRESET.asset_id)
if not norm_stats_path.is_file():
    raise FileNotFoundError(
        f"Missing openpi norm stats: {norm_stats_path}. "
        f"Run: cd ~/openpi && uv run scripts/compute_norm_stats.py "
        f"--config-name {OPENPI_TRAIN_CONFIG_NAME} "
        f'--lerobot-root "{DATASET_ROOT}" '
        f"--output-asset-id {PRESET.asset_id}"
    )
lerobot_stats = openpi_norm_stats_to_lerobot(load_openpi_norm_stats(norm_stats_path))

dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
input_features, output_features = MyPolicy.pi0_features_from_metadata(dataset_metadata)

_effective_batch = PRESET.batch_size * GRAD_ACCUM_STEPS
print(
    f"openpi preset={OPENPI_TRAIN_CONFIG_NAME}  "
    f"horizon={PRESET.action_horizon}  stride={PRESET.action_delta_stride}  "
    f"micro_batch={PRESET.batch_size}  grad_accum={GRAD_ACCUM_STEPS}  "
    f"effective_batch={_effective_batch}  steps={PRESET.num_train_steps}"
)
if resume_ckpt_dir is not None:
    print(
        f"Resume checkpoint: {resume_ckpt_dir} (step {_checkpoint_step(resume_ckpt_dir)})"
    )
    load_model_path = resume_ckpt_dir
    load_model_local_only = True
else:
    print(
        f"Loading PI0 base: {pretrained_model_id} "
        f"(local_files_only={pi_pretrained_local_only})"
    )
    load_model_path = pretrained_model_id
    load_model_local_only = pi_pretrained_local_only
print(f"Norm stats: {norm_stats_path}")
print(f"Dataset: {DATASET_ROOT}")
print(f"Save dir: {SAVE_DIR_PATH}")

# torch.compile + gradient_checkpointing + flash-attn breaks at first forward
# (KeyError: '_scaled_dot_product_flash_attention'). Default: grad_ckpt on, compile off.
# Set PI0_COMPILE_MODEL=1 only on 40GB+ GPUs and accept higher VRAM without grad_ckpt.
COMPILE_PI0_MODEL = os.environ.get("PI0_COMPILE_MODEL", "0") == "1"
USE_GRADIENT_CHECKPOINTING = not COMPILE_PI0_MODEL

cfg = PI0Config(
    input_features=input_features,
    output_features=output_features,
    empty_cameras=PRESET.empty_cameras,
    image_resolution=(224, 224),
    dtype="bfloat16",
    use_amp=True,
    compile_model=COMPILE_PI0_MODEL,
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    freeze_vision_encoder=True,
    train_expert_only=True,
    device=str(device),
)
print(
    f"PI0 config: chunk={cfg.chunk_size}, empty_cameras={cfg.empty_cameras}, "
    f"compile={cfg.compile_model}, grad_ckpt={cfg.gradient_checkpointing}, "
    f"train_expert_only={cfg.train_expert_only}, "
    f"lr={cfg.optimizer_lr}, wd={cfg.optimizer_weight_decay}"
)
if COMPILE_PI0_MODEL:
    print(
        "PI0_COMPILE_MODEL=1: compile on, gradient_checkpointing off (needs ~40GB+ VRAM)."
    )
print(
    "Note: openpi low_mem uses LoRA (gemma_2b_lora + gemma_300m_lora). "
    "LeRobot uses train_expert_only as the closest single-GPU approximation."
)

policy = PI0Policy.from_pretrained(
    str(load_model_path),
    config=cfg,
    dataset_stats=MyPolicy.pi0_dataset_stats(lerobot_stats),
    local_files_only=load_model_local_only,
)

action_key = "actions" if "actions" in dataset_metadata.features else ACTION
action_offsets_sec = action_delta_timestamps_sec(
    dataset_metadata.fps,
    PRESET.action_horizon,
    PRESET.action_delta_stride,
)
delta_timestamps = {action_key: action_offsets_sec}
print(
    f"Action delta timestamps (sec): {action_offsets_sec[:4]} ... ({len(action_offsets_sec)} steps)"
)

policy.train()
policy.to(device)
_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"Model on {device}, trainable params: {_trainable / 1e6:.1f}M")

tokenizer = load_paligemma_tokenizer(allow_hub_download=_allow_hub_tokenizer)
tokenizer.padding_side = "right"
pi_lang = MyPIPolicy(tokenizer, cfg.tokenizer_max_length)

dataset = LeRobotDataset(
    DATASET_REPO,
    delta_timestamps=delta_timestamps,
    root=DATASET_ROOT,
    image_transforms=None,
    video_backend="torchcodec",
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=PRESET.num_workers,
    batch_size=PRESET.batch_size,
    shuffle=True,
    pin_memory=True,
    persistent_workers=PRESET.num_workers > 0,
    drop_last=True,
)
print(
    f"Dataset size: {len(dataset)}, micro-batches/epoch: {len(dataloader)}, "
    f"target optimizer steps={PRESET.num_train_steps}"
)

optimizer_cfg = cfg.get_optimizer_preset()
optimizer = optimizer_cfg.build(filter(lambda p: p.requires_grad, policy.parameters()))
scheduler = cfg.get_scheduler_preset().build(
    optimizer, num_training_steps=PRESET.num_train_steps
)

_global_step = 0
accum_step = 0
if resume_ckpt_dir is not None:
    state_file = resume_ckpt_dir / TRAINING_STATE_FILE
    if state_file.is_file():
        _global_step, accum_step = load_training_state(state_file, optimizer, scheduler)
        print(
            f"Loaded training state from {state_file}: "
            f"step={_global_step}, accum_step={accum_step}"
        )
    else:
        _global_step = _checkpoint_step(resume_ckpt_dir)
        fast_forward_scheduler(scheduler, _global_step)
        print(
            f"Legacy checkpoint (no {TRAINING_STATE_FILE}): "
            f"model weights at step {_global_step}, "
            "optimizer re-init, LR schedule fast-forwarded"
        )
    preset_file = resume_ckpt_dir / "openpi_train_preset.json"
    if preset_file.is_file():
        saved_preset = json.loads(preset_file.read_text())
        if saved_preset.get("dataset_root") != DATASET_ROOT:
            print(
                f"Warning: dataset_root mismatch. checkpoint={saved_preset.get('dataset_root')} "
                f"current={DATASET_ROOT}"
            )

USE_WANDB = 1
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "aubo-i10-fintune")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "phil_ning")
if USE_WANDB:
    wandb_init_kwargs: dict = {
        "project": WANDB_PROJECT,
        "entity": WANDB_ENTITY,
        "name": os.environ.get(
            "WANDB_RUN_NAME", f"{OPENPI_TRAIN_CONFIG_NAME}_{DATASET_NAME}"
        ),
        "dir": str(SAVE_DIR_PATH),
        "config": {
            "openpi_train_config": OPENPI_TRAIN_CONFIG_NAME,
            "asset_id": PRESET.asset_id,
            "action_horizon": PRESET.action_horizon,
            "action_delta_stride": PRESET.action_delta_stride,
            "chunk_size": PRESET.chunk_size,
            "batch_size": PRESET.batch_size,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "effective_batch": _effective_batch,
            "num_train_steps": PRESET.num_train_steps,
            "use_qpos_delta_actions": PRESET.use_qpos_delta_actions,
            "compile_model": cfg.compile_model,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "train_expert_only": cfg.train_expert_only,
            "optimizer_lr": cfg.optimizer_lr,
            "optimizer_weight_decay": cfg.optimizer_weight_decay,
            "dataset_repo": DATASET_REPO,
            "dataset_root": DATASET_ROOT,
            "pretrained_model": pretrained_model_id,
            "norm_stats_path": str(norm_stats_path),
            "save_dir": str(SAVE_DIR_PATH),
            "trainable_params_m": _trainable / 1e6,
            "resume_ckpt": str(resume_ckpt_dir) if resume_ckpt_dir else None,
            "resume_step": _global_step,
        },
    }
    wandb_run_id = os.environ.get("WANDB_RUN_ID")
    if wandb_run_id:
        wandb_init_kwargs["id"] = wandb_run_id
        wandb_init_kwargs["resume"] = "must"
    elif resume_ckpt_dir is not None:
        auto_run_id = find_wandb_run_id(SAVE_DIR_PATH)
        if auto_run_id:
            wandb_init_kwargs["id"] = auto_run_id
            wandb_init_kwargs["resume"] = "must"
    else:
        wandb_init_kwargs["resume"] = "allow"
    wandb.init(**wandb_init_kwargs)
    print(f"WandB initialized: {wandb.run.get_url()}")
else:
    print("WandB disabled (PI0_WANDB=0)")

if _global_step >= PRESET.num_train_steps:
    print(
        f"Already at step {_global_step} >= num_train_steps={PRESET.num_train_steps}. "
        "Nothing to train."
    )
    sys.exit(0)

pbar = tqdm(
    total=PRESET.num_train_steps,
    initial=_global_step,
    desc="Train PI0",
    dynamic_ncols=True,
    leave=True,
)
optimizer.zero_grad(set_to_none=True)

while _global_step < PRESET.num_train_steps:
    for batch in dataloader:
        batch = MyPolicy.move_batch_to_device(batch, device)
        batch = MyPolicy.normalize_pi0_batch(batch)
        if PRESET.use_qpos_delta_actions:
            batch = MyPolicy.apply_qpos_delta_actions(batch)
        batch = MyPolicy.normalize_pi0_training_batch(batch, lerobot_stats)
        batch = pi_lang.inject_language_tokens(batch, device)

        if cfg.compile_model:
            torch.compiler.cudagraph_mark_step_begin()

        loss, _ = policy(batch)
        (loss / GRAD_ACCUM_STEPS).backward()
        accum_step += 1

        if accum_step < GRAD_ACCUM_STEPS:
            continue

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=optimizer_cfg.grad_clip_norm
        )
        optimizer.step()
        scheduler.step()
        _global_step += 1
        accum_step = 0
        optimizer.zero_grad(set_to_none=True)

        _lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(
            step=f"{_global_step}/{PRESET.num_train_steps}",
            loss=f"{loss.item():.4f}",
            lr=f"{_lr:.2e}",
        )
        pbar.update(1)

        if USE_WANDB:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": _lr,
                    "train/step": _global_step,
                },
                step=_global_step,
            )

        if (
            _global_step % PRESET.save_interval == 0
            or _global_step == PRESET.num_train_steps
        ):
            ckpt_dir = SAVE_DIR_PATH / f"checkpoint_{_global_step:06d}"
            policy.save_pretrained(ckpt_dir)
            save_training_state(
                ckpt_dir / TRAINING_STATE_FILE,
                global_step=_global_step,
                accum_step=accum_step,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            (ckpt_dir / "openpi_train_preset.json").write_text(
                json.dumps(
                    {
                        "openpi_train_config": OPENPI_TRAIN_CONFIG_NAME,
                        "asset_id": PRESET.asset_id,
                        "action_horizon": PRESET.action_horizon,
                        "action_delta_stride": PRESET.action_delta_stride,
                        "batch_size": PRESET.batch_size,
                        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
                        "dataset_root": DATASET_ROOT,
                        "norm_stats_path": str(norm_stats_path),
                    },
                    indent=2,
                )
            )
            print(f"Saved checkpoint to {ckpt_dir}")

        if _global_step % PRESET.log_interval == 0:
            print(
                f"Step {_global_step}/{PRESET.num_train_steps}  loss={loss.item():.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if _global_step >= PRESET.num_train_steps:
            break

pbar.close()
policy.save_pretrained(SAVE_DIR_PATH)
save_training_state(
    SAVE_DIR_PATH / TRAINING_STATE_FILE,
    global_step=_global_step,
    accum_step=accum_step,
    optimizer=optimizer,
    scheduler=scheduler,
)
print(f"Training complete. Final weights: {SAVE_DIR_PATH}")
if USE_WANDB:
    wandb.finish()
