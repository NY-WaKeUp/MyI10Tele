"""
Automated check: after multiple MyEnv.reset() calls, cube rotation matrix must vary.
Runs headless (patched init_viewer, no GLFW window).

Run from repo root:
  .venv/bin/python -m pytest tests/test_cube_reset_orientation.py -v
or:
  .venv/bin/python tests/test_cube_reset_orientation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MYSCENE = REPO_ROOT / "assets" / "aubo_i10_inspire" / "myscene.xml"


def _headless_init_viewer(self) -> None:
    """Skip GLFW viewer; only reload simulation state."""
    self.env.reset()


def _collect_cube_R_matrices(n_resets: int, poison_global_rng: bool) -> np.ndarray:
    from src.core.MyEnv import MyEnv

    with patch.object(MyEnv, "init_viewer", _headless_init_viewer):
        env = MyEnv(str(MYSCENE), seed=None)

    mats: list[np.ndarray] = []
    for _ in range(n_resets):
        if poison_global_rng:
            np.random.seed(999_999)
        env.reset(seed=None)
        _, R = env.env.get_pR_body("cube")
        mats.append(np.asarray(R, dtype=np.float64))
    return np.stack(mats, axis=0)


def test_cube_rotation_std_across_resets() -> None:
    """Variance across resets must be visible on rotation entries."""
    Rs = _collect_cube_R_matrices(n_resets=15, poison_global_rng=True)
    assert Rs.shape == (15, 3, 3)
    std_max = float(Rs.std(axis=0).max())
    assert std_max > 0.02, (
        f"cube R entries barely change across resets (std_max={std_max}); "
        "check reset qpos/qvel and layout RNG."
    )


def test_cube_rotations_not_all_identical() -> None:
    """At least two resets must produce materially different rotation matrices."""
    Rs = _collect_cube_R_matrices(n_resets=10, poison_global_rng=True)
    flat = Rs.reshape(len(Rs), -1)
    first = flat[0]
    max_diff = float(np.max(np.linalg.norm(flat - first, axis=1)))
    assert max_diff > 0.05, (
        f"all cube rotations ~identical to first (max fro diff={max_diff}); "
        "orientation randomization or physics settle may be broken."
    )


if __name__ == "__main__":
    test_cube_rotation_std_across_resets()
    test_cube_rotations_not_all_identical()
    print("OK: cube orientation varies across resets (automated check passed).")
