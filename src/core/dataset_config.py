"""Shared paths and action-label settings for auboI10 LeRobot datasets."""

import os

REPO_NAME = "auboI10"
# "qpos": post-step joint (OpenPI-style). "ee_pose": post-step flange xyz+rpy+gripper.
ACTION_LABEL = "ee_pose"  # or "qpos"

PROJECT_DIR = os.path.expanduser("~/MyI10Tele")
XML_PATH = os.path.join(PROJECT_DIR, "assets/aubo_i10_inspire/myscene.xml")
TASK_NAME = "Put cube on the black platform"

AUBOI10_QPOS_ROOT = "~/MyI10Tele/data_auboI10_v2"
AUBOI10_EEPOSE_ROOT = "~/MyI10Tele/data_auboI10_ee_pose"


def dataset_root(label: str | None = None) -> str:
    """LeRobot dataset directory for the given action label."""
    label = label or ACTION_LABEL
    if label == "qpos":
        return os.path.expanduser(AUBOI10_QPOS_ROOT)
    elif label == "ee_pose":
        return os.path.expanduser(AUBOI10_EEPOSE_ROOT)
    else:
        raise ValueError(f"unknown ACTION_LABEL: {label}")


def env_action_type(label: str | None = None) -> str:
    """MyEnv.action_type when executing policy outputs that match dataset actions."""
    return label or ACTION_LABEL
