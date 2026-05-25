#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACT closed-loop rollout in gym_aloha (aligned with official LeRobot eval).

Official eval command (Model Card):
  https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human

Run from repo root with venv active:
  cd src/core && python 4.val_act_aloha_sim.py
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import gym_aloha  # noqa: F401 — registers gym_aloha/AlohaTransferCube-v0
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import AlohaEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.utils import get_safe_torch_device

from core.my_policy import MyPolicy

# --- Previous (custom) dataset / eval settings ---
# DATASET_REPO = "auboI10"
# POLICY_PATH: str | Path | None = None  # auto-find under .ckpt/
# POLICY_PATH = PRETRAINED_MODEL_ID
# N_EPISODES = 50
# BATCH_SIZE = 128

# --- Dataset (local copy of lerobot/aloha_sim_transfer_cube_human) ---
DATASET_REPO = "lerobot/aloha_sim_transfer_cube_human"
DATASET_NAME = "aloha_sim_transfer_cube_human"
DATASET_ROOT = os.path.expanduser(
    f"~/openpi-cache/huggingface/lerobot/lerobot/{DATASET_NAME}"
)

PRETRAINED_MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"
_CKPT_SUBDIR = f"{PRETRAINED_MODEL_ID.split('/')[-1]}/{DATASET_NAME}"

# True: Hub checkpoint from Model Card (~83% on 500 eps). False: local 2.train_act.py ckpt.
USE_OFFICIAL_HUB_CHECKPOINT = True
POLICY_PATH: str | Path | None = (
    PRETRAINED_MODEL_ID if USE_OFFICIAL_HUB_CHECKPOINT else None
)

ALOHA_TASK = "AlohaTransferCube-v0"

MyPolicy.set_visible_cuda_devices("0")
os.environ.setdefault("DISPLAY", ":11.0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEVICE = torch.device("cuda:0")
# Official eval: 500 episodes, batch 50. Set N_EPISODES=10 for a quick smoke test.
N_EPISODES = 500
BATCH_SIZE = 50
SEED = 1000
USE_AMP = False
MAX_VIDEOS = 10
OUTPUT_DIR = Path(f"outputs/eval/act_aloha_sim/{DATASET_NAME}")


def _find_local_ckpt_dir() -> Path | None:
    rel = Path(".ckpt") / _CKPT_SUBDIR
    candidates = [
        Path(__file__).resolve().parent / rel,
        Path(__file__).resolve().parents[2] / rel,
        rel,
    ]
    for p in candidates:
        if (p / "config.json").is_file():
            return p.resolve()
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
        f"No local checkpoint under .ckpt/{_CKPT_SUBDIR}. "
        f"Train with 2.train_act.py or set USE_OFFICIAL_HUB_CHECKPOINT=True."
    )


def _make_eval_processors(
    policy_cfg: PreTrainedConfig,
    policy_path: str | Path,
    device: torch.device,
):
    # --- Previous processor path (local ckpt only, no dataset_stats fallback) ---
    # ckpt = Path(policy_path)
    # if ckpt.is_dir() and (ckpt / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json").is_file():
    #     return make_pre_post_processors(
    #         policy_cfg=policy_cfg,
    #         pretrained_path=str(ckpt),
    #         preprocessor_overrides={"device_processor": {"device": str(device)}},
    #     )
    # print(
    #     "No policy_preprocessor.json in checkpoint; building normalizers from dataset stats "
    #     f"({DATASET_ROOT})."
    # )
    # dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
    # return make_pre_post_processors(
    #     policy_cfg=policy_cfg,
    #     dataset_stats=dataset_metadata.stats,
    #     preprocessor_overrides={"device_processor": {"device": str(device)}},
    # )

    dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
    ckpt = Path(policy_path)
    pretrained_path = str(ckpt.resolve()) if ckpt.is_dir() else str(policy_path)
    if (
        ckpt.is_dir()
        and not (ckpt / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json").is_file()
    ):
        print(
            "No policy_preprocessor.json in local checkpoint; using dataset stats for normalizers."
        )
    return make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )


def main() -> None:
    policy_path = _resolve_policy_path(POLICY_PATH)

    device = get_safe_torch_device(str(DEVICE), log=True)
    print(f"Policy: {policy_path}")
    print(f"Env: gym_aloha/{ALOHA_TASK}  episodes={N_EPISODES}  batch={BATCH_SIZE}")
    print(f"Official hub eval: {USE_OFFICIAL_HUB_CHECKPOINT}")

    env_cfg = AlohaEnv(task=ALOHA_TASK)
    envs = make_env(env_cfg, n_envs=BATCH_SIZE, use_async_envs=False)
    vec_env = envs["aloha"][0]

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = str(device)
    policy_cfg.use_amp = USE_AMP

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = _make_eval_processors(policy_cfg, policy_path, device)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    videos_dir = OUTPUT_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    amp_ctx = torch.autocast(device_type=device.type) if USE_AMP else nullcontext()
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
