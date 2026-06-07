"""Convert between qpos and ee_pose action labels for auboI10 LeRobot datasets."""

from __future__ import annotations

import mujoco
import numpy as np

from utils.MujocoParser import MuJoCoParserClass
from utils.transforms import r2rpy

from core.my_env import gripper_qpos_to_openpi, openpi_gripper_to_rh_r1_ctrl

ARM_JOINT_NAMES = (
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
)
FLANGE_BODY = "i10_inspire_flange_link"
GRIPPER_JOINT = "rh_r1"


class QposToEePoseFK:
    """Headless FK: post-step qpos (7D) -> post-step ee_pose (7D)."""

    def __init__(self, xml_path: str) -> None:
        self._parser = MuJoCoParserClass(
            name="qpos_to_ee_fk", rel_xml_path=xml_path, verbose=False
        )
        self._rh_r1_adr = int(
            self._parser.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self._parser.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT
                )
            ]
        )

    def single(self, qpos7: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos7, dtype=np.float64).reshape(7)
        self._parser.forward(
            q=q[:6], joint_names=list(ARM_JOINT_NAMES), increase_tick=False
        )
        self._parser.data.qpos[self._rh_r1_adr] = openpi_gripper_to_rh_r1_ctrl(q[6])
        mujoco.mj_forward(self._parser.model, self._parser.data)
        p, R = self._parser.get_pR_body(body_name=FLANGE_BODY)
        rpy = r2rpy(R)
        grip = gripper_qpos_to_openpi(float(self._parser.data.qpos[self._rh_r1_adr]))
        return np.array([*p, *rpy, grip], dtype=np.float32)

    def batch(self, qpos_actions: np.ndarray) -> np.ndarray:
        rows = np.asarray(qpos_actions, dtype=np.float64)
        if rows.ndim == 1:
            return self.single(rows)
        out = np.empty((rows.shape[0], 7), dtype=np.float32)
        for i in range(rows.shape[0]):
            out[i] = self.single(rows[i])
        return out
