# -*- coding: utf-8 -*-
"""Shared helpers for LeRobot policy training scripts (ACT, PI0, etc.)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def resolved_hf_hub_cache() -> Path:
    """Hub model cache directory (same layout as Hugging Face Hub on disk).

    Reads ``HF_HUB_CACHE`` if set; else ``HF_HOME/hub``; else ``~/.cache/huggingface/hub``.
    Evaluated at call time so scripts can ``setdefault`` env vars before importing LeRobot.
    """
    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        return Path(env_hub).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser().resolve() / "hub"
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _project_root() -> Path:
    # This file: <project>/src/core/my_policy.py
    return Path(__file__).resolve().parents[2]


def _newest_snapshot_with_file(
    hub: Path,
    models_glob: str,
    required_relative: str,
) -> Path | None:
    """Among ``hub / <match> / snapshots/*``, return the newest dir containing ``required_relative``."""
    best: Path | None = None
    best_mtime = -1
    if not hub.is_dir():
        return None
    for model_dir in hub.glob(models_glob):
        snaps = model_dir / "snapshots"
        if not snaps.is_dir():
            continue
        for p in snaps.iterdir():
            if p.is_dir() and (p / required_relative).exists():
                m = p.stat().st_mtime_ns
                if m > best_mtime:
                    best_mtime = m
                    best = p
    return best


def _newest_snapshot_matching(
    hub: Path,
    models_glob: str,
    predicate: Callable[[Path], bool],
) -> Path | None:
    """Among ``hub / <match> / snapshots/*``, return the newest directory satisfying ``predicate``."""
    best: Path | None = None
    best_mtime = -1
    if not hub.is_dir():
        return None
    for model_dir in hub.glob(models_glob):
        snaps = model_dir / "snapshots"
        if not snaps.is_dir():
            continue
        for p in snaps.iterdir():
            if predicate(p):
                m = p.stat().st_mtime_ns
                if m > best_mtime:
                    best_mtime = m
                    best = p
    return best


def resolve_pi0_pretrained_path(
    explicit_path: str | None = None,
    *,
    allow_hub_download: bool = False,
    preferred_hub_repo: str = "lerobot/pi0_base",
) -> tuple[str, bool]:
    """Pick a local PI0 checkpoint path; avoid Hub downloads unless ``allow_hub_download`` is True.

    Resolution order:
    1. ``explicit_path`` if it contains ``model.safetensors``.
    2. Environment ``PI0_PRETRAINED``.
    3. ``<project>/pretrained/pi0_base`` or ``<project>/pretrained/pi0`` if they contain weights.
    4. Newest snapshot under ``resolved_hf_hub_cache()`` matching ``models--lerobot--pi0*`` with
       ``model.safetensors`` (whichever you already have: pi0_base, pi0, etc.).

    If nothing is found and ``allow_hub_download`` is False, raises ``RuntimeError`` (no silent Hub fallback).

    Returns:
        (pretrained_name_or_path, local_files_only)
    """

    def _has_weights(d: Path) -> bool:
        return d.is_dir() and (d / "model.safetensors").exists()

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    env_p = os.environ.get("PI0_PRETRAINED")
    if env_p:
        candidates.append(Path(env_p).expanduser().resolve())

    root = _project_root()
    for rel in ("pretrained/pi0_base", "pretrained/pi0"):
        candidates.append((root / rel).resolve())

    for c in candidates:
        if _has_weights(c):
            return str(c), True

    hub = resolved_hf_hub_cache()
    snap = _newest_snapshot_with_file(hub, "models--lerobot--pi0*", "model.safetensors")
    if snap is not None:
        return str(snap), True

    if allow_hub_download:
        return preferred_hub_repo, False

    raise RuntimeError(
        "No local PI0 weights found (expected model.safetensors). Checked: PI0_PRETRAINED, "
        f"pretrained/pi0_base, pretrained/pi0, and {hub}/models--lerobot--pi0*/snapshots/*. "
        "Set PI0_PRETRAINED to a snapshot directory, or pass allow_hub_download=True to use the Hub."
    )


def resolve_paligemma_tokenizer_path(
    explicit_path: str | None = None,
    *,
    allow_hub_download: bool = False,
    hub_repo_id: str = "google/paligemma-3b-pt-224",
) -> tuple[str, bool]:
    """Resolve PaliGemma tokenizer from local disk only unless ``allow_hub_download`` is True."""

    def _has_tokenizer(d: Path) -> bool:
        if not d.is_dir():
            return False
        # Transformers resolves tokenizer_config.json first; some snapshots only expose one name.
        return (d / "tokenizer_config.json").exists() or (d / "tokenizer.json").exists()

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    env_p = os.environ.get("PI0_TOKENIZER") or os.environ.get("PALIGEMMA_TOKENIZER")
    if env_p:
        candidates.append(Path(env_p).expanduser().resolve())

    root = _project_root()
    candidates.append((root / "pretrained" / "paligemma-3b-pt-224").resolve())

    for c in candidates:
        if _has_tokenizer(c):
            return str(c), True

    hub = resolved_hf_hub_cache()
    snap = _newest_snapshot_matching(hub, "models--google--paligemma*", _has_tokenizer)
    if snap is not None:
        return str(snap), True

    if allow_hub_download:
        return hub_repo_id, False

    raise RuntimeError(
        "No local PaliGemma tokenizer found (expected tokenizer_config.json or tokenizer.json). "
        "Checked: PI0_TOKENIZER / PALIGEMMA_TOKENIZER, pretrained/paligemma-3b-pt-224, and "
        f"{hub}/models--google--paligemma*/snapshots/*. "
        "Set PI0_TOKENIZER to a snapshot directory, or PI0_TOKENIZER_ALLOW_HUB_DOWNLOAD=1 (needs network)."
    )


class MyPolicy:
    """Base utilities so training scripts do not redefine the same helpers at import time."""

    @staticmethod
    def set_visible_cuda_devices(gpu_index: str) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        os.environ["NVIDIA_VISIBLE_DEVICE"] = gpu_index

    @staticmethod
    def move_batch_to_device(
        batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Move tensors in batch to device; leave strings and non-tensor values unchanged."""
        out: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                out[k] = [x.to(device, non_blocking=True) for x in v]
            else:
                out[k] = v
        return out

    @staticmethod
    def input_output_features_from_metadata(
        dataset_metadata: LeRobotDatasetMetadata,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {
            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
        }
        input_features = {
            key: ft for key, ft in features.items() if key not in output_features
        }
        return input_features, output_features


class MyPIPolicy(MyPolicy):
    """PI0 / PaliGemma language conditioning: tokenize task strings and inject into batch."""

    def __init__(self, tokenizer: Any, tokenizer_max_length: int) -> None:
        self._tokenizer = tokenizer
        self._tokenizer_max_length = tokenizer_max_length

    def tokenize_task_strings(
        self, task_list: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of task strings, appending newline for PaliGemma compatibility."""
        tasks_with_nl = [t if t.endswith("\n") else t + "\n" for t in task_list]
        encoded = self._tokenizer(
            tasks_with_nl,
            return_tensors="pt",
            max_length=self._tokenizer_max_length,
            padding="max_length",
            truncation=True,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def inject_language_tokens(
        self, batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Tokenize task strings and inject language tokens/masks into batch."""
        task_strings = batch.pop("task", None)
        if task_strings is None:
            raise ValueError('Batch missing "task" field for language conditioning')
        if isinstance(task_strings, str):
            task_strings = [task_strings]
        ids, mask = self.tokenize_task_strings(task_strings)
        batch[OBS_LANGUAGE_TOKENS] = ids.to(device, non_blocking=True)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = mask.to(device, non_blocking=True)
        return batch
