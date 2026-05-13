# -*- coding: utf-8 -*-
"""Shared helpers for LeRobot policy training scripts (ACT, PI0, etc.)."""

from __future__ import annotations

import os
from typing import Any

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features


class MyPolicy:
    """Base utilities so training scripts do not redefine the same helpers at import time."""

    @staticmethod
    def set_visible_cuda_devices(gpu_index: str) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        os.environ["NVIDIA_VISIBLE_DEVICE"] = gpu_index

    @staticmethod
    def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
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
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        return input_features, output_features


class MyPIPolicy(MyPolicy):
    """PI0 / PaliGemma language conditioning: tokenize task strings and inject into batch."""

    def __init__(self, tokenizer: Any, tokenizer_max_length: int) -> None:
        self._tokenizer = tokenizer
        self._tokenizer_max_length = tokenizer_max_length

    def tokenize_task_strings(self, task_list: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
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

    def inject_language_tokens(self, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        """Tokenize task strings and inject language tokens/masks into batch."""
        task_strings = batch.pop("task", None)
        if task_strings is None:
            raise ValueError('Batch missing "task" field for language conditioning')
        if isinstance(task_strings, str):
            task_strings = [task_strings]
        ids, mask = self.tokenize_task_strings(task_strings)
        batch["observation.language_tokens"] = ids.to(device, non_blocking=True)
        batch["observation.language_attention_mask"] = mask.to(device, non_blocking=True)
        return batch
