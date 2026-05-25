#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PI0 closed-loop rollout in gym_aloha (pairs with 5.train_pi.py on aloha_sim_transfer_cube_human).

References:
  - https://huggingface.co/docs/lerobot/en/pi0
  - https://github.com/huggingface/lerobot/issues/1951

Run:
  cd src/core && python 7.val_pi_aloha_sim.py
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import gym_aloha  # noqa: F401 — registers gym_aloha/AlohaTransferCube-v0
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.configs import AlohaEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.utils import get_safe_torch_device
from transformers import AutoTokenizer

from core.my_policy import (
    MyPolicy,
    resolve_paligemma_tokenizer_path,
    resolve_pi0_pretrained_path,
)

_hf_root = Path.home() / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_hf_root))
os.environ.setdefault("HF_HUB_CACHE", str(_hf_root / "hub"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# --- Previous (custom) settings ---
# DATASET_REPO = "auboI10"
# N_EPISODES = 50
# BATCH_SIZE = 10
# USE_AMP = True
# COMPILE_PI0_MODEL = False (eval)
# VISIBLE_GPU_INDEX = "1"

# --- Match 5.train_pi.py ---
DATASET_REPO = "lerobot/aloha_sim_transfer_cube_human"
DATASET_NAME = "aloha_sim_transfer_cube_human"
DATASET_ROOT = os.path.expanduser(
    f"~/openpi-cache/huggingface/lerobot/lerobot/{DATASET_NAME}"
)

PI0_PRETRAINED_DIR: str | None = None
_allow_hub = False
_allow_hub_tokenizer = False
_pi_pretrained_id, _ = resolve_pi0_pretrained_path(
    PI0_PRETRAINED_DIR,
    allow_hub_download=_allow_hub,
)

# No Hub checkpoint for Aloha transfer-cube; eval local finetune from 5.train_pi.py.
POLICY_PATH: str | Path | None = None

ALOHA_TASK = "AlohaTransferCube-v0"
VISIBLE_GPU_INDEX = "0"
os.environ.setdefault("DISPLAY", ":11.0")

# Issue #1951 used eval.n_episodes=10 during training; use 50 for quicker check, 500 for full bench.
N_EPISODES = 50
BATCH_SIZE = 10
SEED = 1000
USE_AMP = False
COMPILE_PI0_MODEL = False
MAX_VIDEOS = 10
OUTPUT_DIR = Path(f"outputs/eval/pi0_aloha_sim/{DATASET_NAME}")


def _task_prompt_from_dataset() -> str:
    row = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT)[0]
    task = row["task"]
    if isinstance(task, list):
        task = task[0]
    return str(task)


def _find_local_ckpt_dir() -> Path | None:
    rel_name = DATASET_NAME
    for base in (
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[2],
        Path.cwd(),
    ):
        ckpt_root = base / ".ckpt"
        if not ckpt_root.is_dir():
            continue
        for sub in sorted(ckpt_root.iterdir()):
            ckpt_dir = sub / rel_name
            cfg_path = ckpt_dir / "config.json"
            if not cfg_path.is_file():
                continue
            cfg = json.loads(cfg_path.read_text())
            if cfg.get("type") == "pi0":
                return ckpt_dir.resolve()
    return None


def _resolve_policy_path(path: str | Path | None) -> str | Path:
    if path is not None:
        local = Path(path)
        if local.is_dir() and (local / "config.json").is_file():
            return local.resolve()
        if isinstance(path, str) and "/" in path and not local.exists():
            return path
        raise FileNotFoundError(f"No checkpoint at {local.resolve()}")

    found = _find_local_ckpt_dir()
    if found is not None:
        return found
    raise FileNotFoundError(
        f"No PI0 checkpoint under .ckpt/*/{DATASET_NAME}. Train with 5.train_pi.py first."
    )


def _load_local_paligemma_tokenizer() -> AutoTokenizer:
    tokenizer_src, tokenizer_local = resolve_paligemma_tokenizer_path(
        allow_hub_download=_allow_hub_tokenizer,
    )
    print(f"Tokenizer: {tokenizer_src} (local_files_only={tokenizer_local})")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        local_files_only=tokenizer_local,
    )
    tokenizer.padding_side = "right"
    return tokenizer


def _build_pi0_processors(
    policy_cfg: PI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    tokenizer: AutoTokenizer,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Same step order as make_pi0_pre_post_processors, with local tokenizer (offline)."""
    device_str = str(device)
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        Pi0NewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer=tokenizer,
            max_length=policy_cfg.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=device_str),
        NormalizerProcessorStep(
            features={**policy_cfg.input_features, **policy_cfg.output_features},
            norm_map=policy_cfg.normalization_mapping,
            stats=dataset_stats,
            device=device_str,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=policy_cfg.output_features,
            norm_map=policy_cfg.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _make_eval_processors(
    policy_cfg: PreTrainedConfig,
    policy_path: str | Path,
    device: torch.device,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # --- Previous: always build from dataset stats only ---
    # dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
    # tokenizer = _load_local_paligemma_tokenizer()
    # return _build_pi0_processors(policy_cfg, dataset_metadata.stats, device, tokenizer)

    # --- Prefer lerobot factory (Hub ckpt); fallback to local tokenizer + stats ---
    if not isinstance(policy_cfg, PI0Config):
        raise TypeError(f"Expected PI0Config, got {type(policy_cfg)}")

    dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
    ckpt = Path(policy_path)
    pretrained_path = str(ckpt.resolve()) if ckpt.is_dir() else str(policy_path)
    if ckpt.is_dir() and (ckpt / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json").is_file():
        return make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=pretrained_path,
            dataset_stats=dataset_metadata.stats,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

    print(
        "Building PI0 processors from dataset stats with local PaliGemma tokenizer "
        f"({DATASET_ROOT})."
    )
    tokenizer = _load_local_paligemma_tokenizer()
    return _build_pi0_processors(policy_cfg, dataset_metadata.stats, device, tokenizer)


def _infer_batch_size(obs: dict[str, Any]) -> int:
    task = obs.get("task")
    if isinstance(task, list):
        return len(task)
    for value in obs.values():
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            return int(value.shape[0])
    raise ValueError(
        f"Cannot infer batch size from observation (keys={list(obs.keys())})"
    )


def _wrap_preprocessor_with_task(
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    task_text: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _run(obs: dict[str, Any]) -> dict[str, Any]:
        obs = dict(obs)
        n = _infer_batch_size(obs)
        obs["task"] = [task_text] * n
        return preprocessor(obs)

    return _run


def main() -> None:
    MyPolicy.set_visible_cuda_devices(VISIBLE_GPU_INDEX)
    device = get_safe_torch_device("cuda:0", log=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} -> {device}")

    policy_path = _resolve_policy_path(POLICY_PATH)
    task_text = _task_prompt_from_dataset()
    print(f"Task prompt: {task_text!r}")
    print(f"Policy: {policy_path}")
    print(f"Env: gym_aloha/{ALOHA_TASK}  episodes={N_EPISODES}  batch={BATCH_SIZE}")

    env_cfg = AlohaEnv(task=ALOHA_TASK)
    envs = make_env(env_cfg, n_envs=BATCH_SIZE, use_async_envs=False)
    vec_env = envs["aloha"][0]

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = "cuda:0"
    policy_cfg.use_amp = USE_AMP
    if isinstance(policy_cfg, PI0Config):
        policy_cfg.compile_model = COMPILE_PI0_MODEL
        policy_cfg.gradient_checkpointing = False
        print(
            f"Loaded PI0 eval config: n_action_steps={policy_cfg.n_action_steps}, "
            f"chunk_size={policy_cfg.chunk_size}, empty_cameras={policy_cfg.empty_cameras}"
        )

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = _make_eval_processors(policy_cfg, policy_path, device)
    preprocessor = _wrap_preprocessor_with_task(preprocessor, task_text)

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    videos_dir = OUTPUT_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if USE_AMP
        else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        info = eval_policy(
            env=vec_env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=N_EPISODES,
            max_episodes_rendered=MAX_VIDEOS,
            videos_dir=videos_dir,
            start_seed=SEED,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    agg = info["aggregated"]
    print("-" * 40)
    print(f"Episodes: {N_EPISODES}")
    print(f"Success rate: {agg['pc_success']:.1f}%")
    print(f"Avg sum reward: {agg['avg_sum_reward']:.4f}")
    print(f"Eval time: {agg['eval_s']:.1f}s ({agg['eval_ep_s']:.2f}s / episode)")
    if info.get("video_paths"):
        print(f"Videos: {videos_dir}")
    print(f"Metrics saved: {OUTPUT_DIR / 'eval_info.json'}")
    print("-" * 40)

    vec_env.close()


if __name__ == "__main__":
    main()
