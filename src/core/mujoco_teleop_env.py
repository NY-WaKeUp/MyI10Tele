import os
from dataclasses import dataclass

from dm_control import mujoco as dm_mujoco
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.dataset.task_spec import TaskSpec


AUBO_I10_HOME_ARM_RAD = np.array(
    [-1.0568138360977173, -0.48808249831199646, 1.3184903860092163, 0.22961488366127014, 1.5413566827774048, 0.5091112852096558],
    dtype=np.float64,
)


@dataclass
class StepOutput:
    state: np.ndarray
    action: np.ndarray
    done: bool
    success: bool


class MujocoTeleopEnv:
    def __init__(self, model_path: str, seed: int, ik_damping: float, ik_gain: float) -> None:
        self.physics = dm_mujoco.Physics.from_xml_path(model_path)
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.rng = np.random.default_rng(seed)
        self.ik_damping = ik_damping
        self.ik_gain = ik_gain

        self.arm_joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]
        self.arm_joint_ids = np.array([self._joint_id(n) for n in self.arm_joint_names], dtype=np.int32)
        self.arm_qpos_idx = np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids], dtype=np.int32)
        self.arm_dof_idx = np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids], dtype=np.int32)
        self.arm_ctrl_idx = np.array([self._act_id(f"{n}_servo") for n in self.arm_joint_names], dtype=np.int32)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.gripper_joint_id = self._joint_id("rh_r1")
        self.gripper_qpos_idx = self.model.jnt_qposadr[self.gripper_joint_id]
        self.gripper_ctrl_idx = self._act_id("rh_r1_servo")

        self.cube_qpos_adr = self._cube_free_joint_qpos_adr()
        self.cube_spawn_z = float(self.model.qpos0[self.cube_qpos_adr + 2])

        self.place_delta_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "place_target_delta")
        assert self.place_delta_site_id >= 0, "site 'place_target_delta' not found on cube (expected in myscene.xml)"

        self.place_target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "place_target_platform")
        assert self.place_target_body_id >= 0, "body 'place_target_platform' not found (expected in myscene.xml)"
        self.place_target_mocapid = int(self.model.body_mocapid[self.place_target_body_id])
        assert self.place_target_mocapid >= 0, "place_target_platform must be mocap body (mocap='true' in MJCF) for per-episode pose"

        deck_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "place_target_deck")
        assert deck_gid >= 0, "geom 'place_target_deck' not found"
        gs = np.asarray(self.model.geom_size[deck_gid], dtype=np.float64)
        self.place_deck_half_z = float(gs[2])
        # Horizontal tolerance scales with deck footprint (no duplicate MJCF lengths here).
        self.place_tol_xy = float(0.75 * min(gs[0], gs[1]))
        # Vertical slack scales with deck thickness.
        self.place_height_eps = float(0.25 * self.place_deck_half_z)
        self.place_platform_z = float(np.asarray(self.model.body_pos[self.place_target_body_id])[2])

        self.agentview_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
        assert self.agentview_cam_id >= 0, "camera 'agentview' not found in model (expected from table.xml for global view)"

        self.wrist_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        assert self.wrist_cam_id >= 0, "camera 'wrist_cam' not found in model (expected on gripper_base_link in aubo_i10_inspire.xml)"

        self.target_pos = np.zeros(3, dtype=np.float64)
        self.target_rot = np.eye(3, dtype=np.float64)
        self.gripper_target = 0.0
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._cam_global = self.global_camera()
        self._cam_wrist = self.wrist_camera()

    def _joint_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _act_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _cube_free_joint_qpos_adr(self) -> int:
        for j in range(self.model.njnt):
            if int(self.model.jnt_bodyid[j]) == int(self.cube_body_id):
                return int(self.model.jnt_qposadr[j])
        assert False, "no joint found on body 'cube'"

    def reset(self, task: TaskSpec) -> None:
        self.physics.reset()
        data = self.physics.data
        data.qpos[self.arm_qpos_idx] = AUBO_I10_HOME_ARM_RAD
        data.qvel[:] = 0.0
        data.ctrl[self.arm_ctrl_idx] = AUBO_I10_HOME_ARM_RAD
        data.qpos[self.gripper_qpos_idx] = 0.0
        data.ctrl[self.gripper_ctrl_idx] = 0.0

        cube_xy = self.rng.uniform(task.cube_xy_low, task.cube_xy_high)
        cube_q = data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 7]
        cube_q[:2] = cube_xy
        cube_q[2] = self.cube_spawn_z
        cube_q[3:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        off_b = np.asarray(self.model.site_pos[self.place_delta_site_id], dtype=np.float64)
        q_wxyz = cube_q[3:7]
        rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        delta_w = rot.apply(off_b)
        mid = self.place_target_mocapid
        data.mocap_pos[3 * mid : 3 * mid + 3] = np.array(
            [float(cube_xy[0]) + float(delta_w[0]), float(cube_xy[1]) + float(delta_w[1]), self.place_platform_z],
            dtype=np.float64,
        )
        data.mocap_quat[4 * mid : 4 * mid + 4] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)

        self.target_pos = data.site_xpos[self.ee_site_id].copy()
        self.target_rot = data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        self.gripper_target = float(data.qpos[self.gripper_qpos_idx])

    def get_state(self) -> np.ndarray:
        return np.concatenate(
            [self.data.qpos[self.arm_qpos_idx].astype(np.float32), np.array([self.data.qpos[self.gripper_qpos_idx]], dtype=np.float32)]
        )

    def _rotation_error(self, current_mat: np.ndarray, target_mat: np.ndarray) -> np.ndarray:
        return (R.from_matrix(target_mat) * R.from_matrix(current_mat).inv()).as_rotvec()

    def _solve_ik_step(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        pos_err = self.target_pos - self.data.site_xpos[self.ee_site_id]
        rot_err = self._rotation_error(self.data.site_xmat[self.ee_site_id].reshape(3, 3), self.target_rot)
        err6 = np.concatenate([pos_err, rot_err])
        jac6 = np.vstack([jacp[:, self.arm_dof_idx], jacr[:, self.arm_dof_idx]])
        lhs = jac6.T @ jac6 + self.ik_damping * np.eye(self.arm_dof_idx.size, dtype=np.float64)
        rhs = jac6.T @ err6
        return self.ik_gain * np.linalg.solve(lhs, rhs)

    def step(self, action: np.ndarray, sim_steps_per_control: int, task: TaskSpec) -> StepOutput:
        self.target_pos = self.target_pos + action[:3].astype(np.float64)
        self.target_rot = R.from_rotvec(action[3:6].astype(np.float64)).as_matrix() @ self.target_rot

        low, high = self.model.jnt_range[self.gripper_joint_id]
        self.gripper_target = float(np.clip(self.gripper_target + float(action[6]), low, high))

        dq = self._solve_ik_step()
        self.physics.data.ctrl[self.arm_ctrl_idx] = self.physics.data.qpos[self.arm_qpos_idx] + dq
        self.physics.data.ctrl[self.gripper_ctrl_idx] = self.gripper_target
        self.physics.step(nstep=sim_steps_per_control)

        success = self._check_success(task)
        return StepOutput(state=self.get_state(), action=action.astype(np.float32), done=success, success=success)

    def _check_success(self, task: TaskSpec) -> bool:
        ee = self.data.site_xpos[self.ee_site_id]
        cube = self.data.xpos[self.cube_body_id]
        if task.task_id == "reach":
            return np.linalg.norm(ee - (cube + np.array([0.0, 0.0, 0.12], dtype=np.float64))) < 0.04
        if task.task_id == "grasp":
            close_enough = np.linalg.norm(ee - cube) < 0.06
            gripper_closed = self.data.qpos[self.gripper_qpos_idx] > 0.45
            cube_height_up = cube[2] > 0.27
            return bool(close_enough and gripper_closed and cube_height_up)
        ped = self.data.xpos[self.place_target_body_id]
        cube_on_target = bool(
            np.linalg.norm(cube[:2] - ped[:2]) < self.place_tol_xy
            and cube[2] > ped[2] + self.place_deck_half_z - self.place_height_eps
        )
        gripper_open = self.data.qpos[self.gripper_qpos_idx] < 0.15
        return bool(cube_on_target and gripper_open)

    def global_camera(self) -> mujoco.MjvCamera:
        """Fixed scene camera `agentview` from MJCF (table.xml): third-person / workspace view."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self.agentview_cam_id
        return cam

    def wrist_camera(self) -> mujoco.MjvCamera:
        """Gripper-mounted camera `wrist_cam` from MJCF (rigid child of gripper_base_link; top-down toward fingers)."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self.wrist_cam_id
        return cam

    def render_main(self, width: int, height: int) -> np.ndarray:
        renderer = self._get_renderer(width=width, height=height)
        renderer.update_scene(self.data, camera=self._cam_global)
        return renderer.render()

    def render_wrist(self, width: int, height: int) -> np.ndarray:
        renderer = self._get_renderer(width=width, height=height)
        renderer.update_scene(self.data, camera=self._cam_wrist)
        return renderer.render()

    def _get_renderer(self, width: int, height: int) -> mujoco.Renderer:
        key = (int(width), int(height))
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, width=key[0], height=key[1])
            self._renderers[key] = renderer
        return renderer


def default_model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "assets", "aubo_i10_inspire", "myscene.xml"))
