"""Convert between qpos and ee_pose action labels for auboI10 LeRobot datasets."""

from __future__ import annotations

import mujoco
import numpy as np

from utils.MujocoParser import MuJoCoParserClass
from utils.transforms import r2rpy
from utils.utils import rpy2r, solve_ik

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


class EePoseToQposIK:
    """Headless IK: post-step ee_pose (7D) -> post-step qpos (7D).

    Uses pre-step arm joints as q_init (``observation.state[:6]``), matching
    MyEnv absolute ee_pose execution. IK is redundant: recovered qpos may differ
    from the original teleop solution when multiple joint configs reach the same EE.
    """

    def __init__(self, xml_path: str) -> None:
        self._parser = MuJoCoParserClass(
            name="ee_to_qpos_ik", rel_xml_path=xml_path, verbose=False
        )

    def single(
        self, ee_pose7: np.ndarray, q_init6: np.ndarray
    ) -> tuple[np.ndarray, float]:
        ee = np.asarray(ee_pose7, dtype=np.float64).reshape(7)
        q0 = np.asarray(q_init6, dtype=np.float64).reshape(6)
        p_trgt = ee[:3]
        R_trgt = rpy2r(ee[3:6])
        q_arm, ik_err_stack, _ = solve_ik(
            env=self._parser,
            joint_names_for_ik=list(ARM_JOINT_NAMES),
            body_name_trgt=FLANGE_BODY,
            q_init=q0,
            p_trgt=p_trgt,
            R_trgt=R_trgt,
            max_ik_tick=50,
            ik_stepsize=1.0,
            ik_eps=1e-2,
            ik_th=np.radians(5.0),
            restore_state=True,
            render=False,
            verbose_warning=False,
        )
        ik_err = float(np.linalg.norm(ik_err_stack))
        qpos7 = np.array([*q_arm, ee[6]], dtype=np.float32)
        return qpos7, ik_err

    def episode(
        self,
        ee_actions: np.ndarray,
        pre_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ee_actions = np.asarray(ee_actions, dtype=np.float64)
        pre_states = np.asarray(pre_states, dtype=np.float64)
        n = ee_actions.shape[0]
        out = np.empty((n, 7), dtype=np.float32)
        ik_errs = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i], ik_errs[i] = self.single(ee_actions[i], pre_states[i, :6])
        return out, ik_errs
