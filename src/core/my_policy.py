# -*- coding: utf-8 -*-
"""Shared helpers for LeRobot policy training scripts (ACT, PI0, etc.)."""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


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


def _core_dir() -> Path:
    return Path(__file__).resolve().parent


def openpi_data_home() -> Path:
    """Same default as ``openpi.shared.download.get_cache_dir`` when ``OPENPI_DATA_HOME`` is set."""
    return (
        Path(os.environ.get("OPENPI_DATA_HOME", str(Path.home() / "openpi-cache")))
        .expanduser()
        .resolve()
    )


def openpi_cached_path(url: str) -> Path:
    """Local path for an openpi ``gs://`` asset (mirrors ``download.maybe_download``, no network)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "":
        return Path(url).expanduser().resolve()
    return openpi_data_home() / parsed.netloc / parsed.path.strip("/")


# openpi TrainConfig uses gs://openpi-assets/checkpoints/pi0_base/params (JAX / Orbax, not LeRobot).
OPENPI_PI0_BASE_PARAMS = openpi_cached_path(
    "gs://openpi-assets/checkpoints/pi0_base/params"
)
OPENPI_PALIGEMMA_TOKENIZER_MODEL = openpi_cached_path(
    "gs://big_vision/paligemma_tokenizer.model"
)


def openpi_repo_root() -> Path:
    """Local openpi checkout (``OPENPI_ROOT`` or ``~/openpi``)."""
    return Path(os.environ.get("OPENPI_ROOT", Path.home() / "openpi")).expanduser().resolve()


def openpi_norm_stats_path(train_config_name: str, asset_id: str) -> Path:
    return openpi_repo_root() / "assets" / train_config_name / asset_id / "norm_stats.json"


def load_openpi_norm_stats(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(path.read_text())
    return payload["norm_stats"]


def openpi_norm_stats_to_lerobot(
    norm_stats: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    """Map openpi ``state`` / ``actions`` keys to LeRobot PI0 feature keys."""
    out: dict[str, dict[str, list[float]]] = {}
    if "state" in norm_stats:
        out["observation.state"] = norm_stats["state"]
    if "actions" in norm_stats:
        out["action"] = norm_stats["actions"]
    return out


def action_delta_timestamps_sec(
    fps: float,
    action_horizon: int,
    action_delta_stride: int,
) -> list[float]:
    """Same formula as ``openpi.training.data_loader.action_delta_timestamps_sec``."""
    if action_delta_stride <= 1:
        frame_offsets = list(range(action_horizon))
    else:
        frame_offsets = [action_delta_stride * (t + 1) for t in range(action_horizon)]
    return [offset / fps for offset in frame_offsets]


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
    """Pick a local LeRobot PI0 checkpoint (``model.safetensors``).

    openpi's ``pi0_base`` lives at ``OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params``
    as JAX/Orbax weights (see ``openpi.training.weight_loaders.CheckpointWeightLoader``).
    LeRobot ``PI0Policy.from_pretrained`` needs a PyTorch dir with ``model.safetensors``.

    Resolution order (same priority as Aserver):
    1. ``explicit_path`` if it contains ``model.safetensors``.
    2. Environment ``PI0_PRETRAINED``.
    3. ``<project>/pretrained/pi0_base`` or ``pretrained/pi0``.
    4. Newest HF Hub snapshot under ``models--lerobot--pi0*``.
    5. ``<project>/src/core/.ckpt/auboI10_pi0_finetuned`` (local fine-tune fallback).

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

    finetuned = (_core_dir() / ".ckpt" / "auboI10_pi0_finetuned").resolve()
    if _has_weights(finetuned):
        return str(finetuned), True

    if allow_hub_download:
        return preferred_hub_repo, False

    openpi_jax = OPENPI_PI0_BASE_PARAMS
    raise RuntimeError(
        "No local LeRobot PI0 weights (model.safetensors). Checked: PI0_PRETRAINED, "
        "pretrained/pi0_base, pretrained/pi0, src/core/.ckpt/auboI10_pi0_finetuned, "
        f"and {hub}/models--lerobot--pi0*/snapshots/*. "
        f"openpi pi0_base JAX weights exist at {openpi_jax} but are not LeRobot format; "
        "set PI0_PRETRAINED to a LeRobot checkpoint dir, or pass allow_hub_download=True."
    )


def _has_hf_tokenizer_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    return (d / "tokenizer_config.json").exists() or (d / "tokenizer.json").exists()


def resolve_paligemma_tokenizer_path(
    explicit_path: str | None = None,
    *,
    allow_hub_download: bool = False,
    hub_repo_id: str = "google/paligemma-3b-pt-224",
) -> tuple[str, bool]:
    """Resolve a HuggingFace tokenizer directory, if present locally."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    env_p = os.environ.get("PI0_TOKENIZER") or os.environ.get("PALIGEMMA_TOKENIZER")
    if env_p:
        candidates.append(Path(env_p).expanduser().resolve())

    root = _project_root()
    candidates.append((root / "pretrained" / "paligemma-3b-pt-224").resolve())

    for c in candidates:
        if _has_hf_tokenizer_dir(c):
            return str(c), True

    hub = resolved_hf_hub_cache()
    snap = _newest_snapshot_matching(
        hub, "models--google--paligemma*", _has_hf_tokenizer_dir
    )
    if snap is not None:
        return str(snap), True

    if allow_hub_download:
        return hub_repo_id, False

    openpi_tok = OPENPI_PALIGEMMA_TOKENIZER_MODEL
    raise RuntimeError(
        "No HuggingFace PaliGemma tokenizer dir found. Checked: PI0_TOKENIZER, "
        f"pretrained/paligemma-3b-pt-224, and {hub}/models--google--paligemma*/snapshots/*. "
        f"Use load_paligemma_tokenizer() to load openpi's local file at {openpi_tok}, "
        "or pass allow_hub_download=True."
    )


def load_paligemma_tokenizer(
    explicit_path: str | None = None,
    *,
    allow_hub_download: bool = False,
    hub_repo_id: str = "google/paligemma-3b-pt-224",
) -> Any:
    """Load PaliGemma tokenizer: HF dir if cached, else openpi ``big_vision/paligemma_tokenizer.model``."""
    from transformers import AutoTokenizer, GemmaTokenizer

    try:
        src, local_only = resolve_paligemma_tokenizer_path(
            explicit_path,
            allow_hub_download=allow_hub_download,
            hub_repo_id=hub_repo_id,
        )
        return AutoTokenizer.from_pretrained(src, local_files_only=local_only)
    except RuntimeError:
        if allow_hub_download:
            raise

    sp_path = OPENPI_PALIGEMMA_TOKENIZER_MODEL
    if explicit_path:
        sp_path = Path(explicit_path).expanduser().resolve()
    if not sp_path.is_file():
        raise RuntimeError(
            f"PaliGemma tokenizer not found. Expected openpi cache file: {OPENPI_PALIGEMMA_TOKENIZER_MODEL} "
            f"(same as openpi.models.tokenizer.PaligemmaTokenizer via gs://big_vision/paligemma_tokenizer.model)."
        )
    print(f"Tokenizer: {sp_path} (openpi SentencePiece, local)")
    return GemmaTokenizer(vocab_file=str(sp_path))


def _dataset_action_key(dataset_metadata: LeRobotDatasetMetadata) -> str:
    """LeRobot v3 datasets use ``actions``; PI0 policy expects ``action`` in the batch."""
    if "actions" in dataset_metadata.features:
        return "actions"
    if ACTION in dataset_metadata.features:
        return ACTION
    raise RuntimeError(
        f"Dataset has no action column (expected 'actions' or '{ACTION}'): "
        f"{list(dataset_metadata.features.keys())}"
    )


def _rename_action_feature_key(features: dict[str, Any]) -> dict[str, Any]:
    if "actions" in features and ACTION not in features:
        features = dict(features)
        features[ACTION] = features.pop("actions")
    return features


def _rename_action_stats(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if stats is None:
        return None
    if "actions" in stats and ACTION not in stats:
        stats = dict(stats)
        stats[ACTION] = stats.pop("actions")
    return stats


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

    @staticmethod
    def pi0_features_from_metadata(
        dataset_metadata: LeRobotDatasetMetadata,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """PI0 config expects output feature key ``action``, not v3 ``actions``."""
        input_features, output_features = MyPolicy.input_output_features_from_metadata(
            dataset_metadata
        )
        return input_features, _rename_action_feature_key(output_features)

    @staticmethod
    def pi0_dataset_stats(stats: dict[str, Any] | None) -> dict[str, Any] | None:
        return _rename_action_stats(stats)

    @staticmethod
    def resolve_pi0_delta_timestamps(
        cfg: Any, dataset_metadata: LeRobotDatasetMetadata
    ) -> dict[str, list[float]] | None:
        """Build delta timestamps using the dataset's actual action column name."""
        if cfg.action_delta_indices is None:
            return None
        action_key = _dataset_action_key(dataset_metadata)
        return {
            action_key: [i / dataset_metadata.fps for i in cfg.action_delta_indices]
        }

    @staticmethod
    def normalize_pi0_batch(batch: dict[str, Any]) -> dict[str, Any]:
        """Map v3 ``actions`` batches to PI0's ``action`` key."""
        if "actions" in batch and ACTION not in batch:
            batch = dict(batch)
            batch[ACTION] = batch.pop("actions")
        return batch

    @staticmethod
    def apply_qpos_delta_actions(batch: dict[str, Any]) -> dict[str, Any]:
        """Match openpi ``DeltaActions(make_bool_mask(6, -1))`` on arm dims only."""
        batch = MyPolicy.normalize_pi0_batch(batch)
        state = batch["observation.state"]
        actions = batch[ACTION]
        delta = actions.clone()
        delta[..., :6] = actions[..., :6] - state[..., :6].unsqueeze(-2)
        batch = dict(batch)
        batch[ACTION] = delta
        return batch

    @staticmethod
    def normalize_pi0_training_batch(
        batch: dict[str, Any],
        stats: dict[str, dict[str, list[float] | torch.Tensor]],
        *,
        eps: float = 1e-8,
    ) -> dict[str, Any]:
        """Mean/std normalize state and action using openpi-computed stats."""
        batch = dict(batch)
        for key in ("observation.state", ACTION):
            if key not in batch or key not in stats:
                continue
            mean = torch.as_tensor(
                stats[key]["mean"], device=batch[key].device, dtype=batch[key].dtype
            )
            std = torch.as_tensor(
                stats[key]["std"], device=batch[key].device, dtype=batch[key].dtype
            )
            x = batch[key]
            if x.ndim == 3:
                mean = mean.view(1, 1, -1)
                std = std.view(1, 1, -1)
            else:
                mean = mean.view(1, -1)
                std = std.view(1, -1)
            batch[key] = (x - mean) / (std + eps)
        return batch


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
        # HF attention_mask is int64; PI0 make_att_2d_masks needs bool pad_masks.
        batch[OBS_LANGUAGE_ATTENTION_MASK] = mask.to(device, non_blocking=True).bool()
        return batch
