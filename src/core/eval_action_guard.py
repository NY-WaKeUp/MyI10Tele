"""Clamp policy actions at eval time to match teleop-scale motion and keep EE over the table."""

from __future__ import annotations

import numpy as np

# front_object_table geom: pos (0,-1,0.115), half (1.0, 0.2, 0.115) -> top z=0.23
TABLE_TOP_Z = 0.23
TABLE_Y_MIN = -1.2
TABLE_Y_MAX = -0.8

# Safe EE workspace over the pick table (meters, world frame).
EE_WORKSPACE_XYZ_MIN = np.array([0.20, -1.11, 0.32], dtype=np.float64)
EE_WORKSPACE_XYZ_MAX = np.array([0.40, -1.02, 0.52], dtype=np.float64)

# Cube still on table if center stays inside these bounds (conservative).
CUBE_Z_MIN_ON_TABLE = TABLE_TOP_Z + 0.012
CUBE_Y_MIN_ON_TABLE = TABLE_Y_MIN + 0.02
CUBE_Y_MAX_ON_TABLE = TABLE_Y_MAX - 0.02


def wrap_pi(angles: np.ndarray) -> np.ndarray:
    return ((np.asarray(angles, dtype=np.float64) + np.pi) % (2 * np.pi)) - np.pi


def clamp_absolute_ee_action(
    action: np.ndarray,
    ee_pre: np.ndarray,
    *,
    max_xyz_step: float = 0.005,
    max_rpy_step: float = 0.12,
    xyz_min: np.ndarray = EE_WORKSPACE_XYZ_MIN,
    xyz_max: np.ndarray = EE_WORKSPACE_XYZ_MAX,
) -> tuple[np.ndarray, bool]:
    """
    Limit per-step absolute EE targets: cap delta from ee_pre, then clip xyz workspace.
    Returns (clamped_action, was_modified).
    """
    a = np.asarray(action, dtype=np.float64).copy()
    pre = np.asarray(ee_pre, dtype=np.float64)
    modified = False

    dxyz = a[:3] - pre[:3]
    dn = float(np.linalg.norm(dxyz))
    if dn > max_xyz_step:
        a[:3] = pre[:3] + dxyz * (max_xyz_step / dn)
        modified = True

    drpy = wrap_pi(a[3:6] - pre[3:6])
    drn = float(np.linalg.norm(drpy))
    if drn > max_rpy_step:
        a[3:6] = pre[3:6] + drpy * (max_rpy_step / drn)
        modified = True

    clipped = np.clip(a[:3], xyz_min, xyz_max)
    if not np.allclose(clipped, a[:3]):
        a[:3] = clipped
        modified = True

    return a, modified


def cube_pose(env) -> np.ndarray:
    return np.asarray(env.env.get_p_body("cube"), dtype=np.float64)


def cube_on_table(p_cube: np.ndarray) -> bool:
    return bool(
        p_cube[2] >= CUBE_Z_MIN_ON_TABLE
        and CUBE_Y_MIN_ON_TABLE <= p_cube[1] <= CUBE_Y_MAX_ON_TABLE
    )
