"""Val-side observation builders.

Two sources, both must match what ``policy.infer`` / training dataloader expect **before**
server transforms (Normalize + ResizeImages 224 happen inside ``serve_policy``):

1. **sim** — same 20Hz path as ``0.tele.py``: pre_state → grab → cv2 256 INTER_AREA
2. **dataset** — LeRobot decoded frame (256 uint8 HWC as on disk), via ``lerobot_item_to_policy_obs``

Do not import this module from ``0.tele.py``.
"""

from __future__ import annotations

import numpy as np

TELE_IMAGE_SIZE = 256


def preprocess_lerobot_image(img: np.ndarray) -> np.ndarray:
    """Mirror 0.tele.py: cv2.resize(..., (256, 256), interpolation=cv2.INTER_AREA)."""
    import cv2

    return cv2.resize(np.asarray(img), (TELE_IMAGE_SIZE, TELE_IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def build_openpi_observation_from_sim(
    agent_img_raw: np.ndarray,
    wrist_img_raw: np.ndarray,
    pre_state: np.ndarray,
    prompt: str,
) -> dict:
    """Sim camera → tele-style 256 tensors → openpi slash-key dict."""
    agent_img = preprocess_lerobot_image(agent_img_raw)
    wrist_img = preprocess_lerobot_image(wrist_img_raw)
    return {
        "observation/image": agent_img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.asarray(pre_state, dtype=np.float32),
        "prompt": prompt,
    }


def build_openpi_observation_from_lerobot_item(
    item: dict,
    tasks: dict[int, str],
    default_prompt: str,
) -> dict:
    """LeRobot row — identical to training after repack (uses openpi if available)."""
    from openpi.policies.aubo_policy import lerobot_item_to_policy_obs

    return lerobot_item_to_policy_obs(item, tasks, default_prompt)
