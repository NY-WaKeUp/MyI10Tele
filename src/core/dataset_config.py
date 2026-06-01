"""Shared paths and action-label settings for auboI10 LeRobot datasets."""

import os

REPO_NAME = "auboI10"
# "qpos": post-step joint targets (OpenPI-style).
# "ee_pose": post-step flange xyz+rpy+gripper (observation.state stays qpos).
ACTION_LABEL = "qpos"  # or "ee_pose"

PROJECT_DIR = os.path.expanduser("~/MyI10Tele")
XML_PATH = os.path.join(PROJECT_DIR, "assets/aubo_i10_inspire/myscene.xml")
TASK_NAME = "Put cube on the black platform"

# LeRobot dataset directories (v2.0 layout with episode_*.parquet).
AUBOI10_QPOS_ROOT = "~/MyI10Tele/data_auboI10_qpos_v20"
AUBOI10_QPOS_ROOT_CONTINUOUS = "~/MyI10Tele/data_auboI10_qpos_v30_continuous"
# obj_init shape (10,): cube_xyz + cube_quat + platform_xyz (see my_env.SCENE_LAYOUT_DIM)
# Interpolated / densified copy (scripts/densify_lerobot_dataset.py); smoother qpos at 20 Hz.
AUBOI10_QPOS_ROOT_INTERP = "~/MyI10Tele/data_auboI10_qpos_v21_interp"
AUBOI10_EEPOSE_ROOT = "~/MyI10Tele/data_auboI10_ee_pose_v21_continuous"

# openpi TrainConfig names (see openpi/src/openpi/training/config.py).
OPENPI_TRAIN_CONFIG_QPOS = "pi0_auboI10_low_mem_finetune_qpos"
OPENPI_TRAIN_CONFIG_EE_POSE = "pi0_auboI10_low_mem_finetune_ee_pose"


def dataset_root(label: str | None = None, *, interp: bool = False) -> str:
    """LeRobot dataset directory for the given action label."""
    label = label or ACTION_LABEL
    if label == "qpos":
        if interp:
            return os.path.expanduser(AUBOI10_QPOS_ROOT_INTERP)
        return os.path.expanduser(AUBOI10_QPOS_ROOT_CONTINUOUS)
    if label == "ee_pose":
        return os.path.expanduser(AUBOI10_EEPOSE_ROOT)
    raise ValueError(f"unknown ACTION_LABEL: {label}")


def env_action_type(label: str | None = None) -> str:
    """MyEnv.action_type when executing policy outputs that match dataset actions."""
    return label or ACTION_LABEL


def policy_ee_pose_command(action_type: str | None = None) -> str:
    """MyEnv.ee_pose_command when running a trained policy (not keyboard teleop)."""
    action_type = action_type or env_action_type()
    if action_type == "ee_pose":
        return "absolute"
    return "delta"


def openpi_train_config_name(label: str | None = None) -> str:
    label = label or ACTION_LABEL
    if label == "qpos":
        return OPENPI_TRAIN_CONFIG_QPOS
    if label == "ee_pose":
        return OPENPI_TRAIN_CONFIG_EE_POSE
    raise ValueError(f"unknown ACTION_LABEL: {label}")
