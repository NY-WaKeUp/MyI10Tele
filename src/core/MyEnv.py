import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable so top-level `utils` package resolves in notebooks and scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.MujocoParser import MuJoCoParserClass
from utils.utils import rotation_matrix,sample_xyzs,rpy2r,add_title_to_img,solve_ik
from utils.transforms import r2quat, r2rpy
import glfw
import copy
import mujoco
from dataclasses import dataclass


class TypedNdArray(np.ndarray):
    """
    A numpy ndarray subclass carrying a lightweight semantic tag in `.type`.

    This keeps full ndarray behavior (shape/dtype/serialization) while enabling
    convenient runtime type checks in notebooks and data pipelines.
    """

    type: str

    def __new__(cls, input_array, type: str):
        obj = np.asarray(input_array).view(cls)
        obj.type = type
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.type = getattr(obj, "type", None)


def as_typed(x: np.ndarray, type: str) -> TypedNdArray:
    return TypedNdArray(x, type=type)

HOME_ARM_QPOS = np.array(
    [
        -1.0568138360977173,
        -0.48808249831199646,
        1.3184903860092163,
        0.22961488366127014,
        1.5413566827774048,
        0.5091112852096558,
    ],
    dtype=np.float64,
)

# Reset orientation target: make flange face vertically down.
# A simple choice is roll=pi, pitch=0, yaw=0 (keeps x-axis along world +x).
RESET_EE_RPY_RAD = np.array([np.pi, 0.0, 0.0], dtype=np.float64)

# Default cube pose from assets/aubo_i10_inspire/myscene.xml <body name="cube" pos="...">.
CUBE_SPAWN_XYZ = np.array([0.2711248850767859, -1.1, 0.255], dtype=np.float64)
# Randomize around that pose (meters). z fixed on table; tighten/loosen as needed.
CUBE_SAMPLE_DX = 0.08
CUBE_SAMPLE_DY = 0.05
CUBE_SAMPLE_DZ = 0.0

class MyEnv:
    def __init__(
        self,
        xml_path: str | None = None,
        action_type: str = "ee_pose",  # or 'qpos' or 'delta_qpos'
        state_type: str = "qpos",  # or 'ee_pose' or 'delta_qpos'
        seed: int = 0,
        ik_damping: float = 1e-3,
        ik_gain: float = 0.6,
    ) -> None:
        model_path = xml_path 
        self.env=MuJoCoParserClass(name="myenv", rel_xml_path=model_path)
        self.action_type = action_type
        self.state_type = state_type
        self.joint_names = ['shoulder_joint',
                    'upperArm_joint',
                    'foreArm_joint',
                    'wrist1_joint',
                    'wrist2_joint',
                    'wrist3_joint',]
        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        # Match collect_vla_dataset.py (viewer.launch_passive camera block).
        self.env.init_viewer(
            azimuth           = -10,
            elevation         = -10,
            distance          = 2.0,
            lookat            = np.array([0.4, -1.0, 0.43], dtype=np.float64),
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed != None: np.random.seed(seed=0) 
        # Compute EE position from HOME_ARM_QPOS by forward kinematics.
        q_init = HOME_ARM_QPOS.copy()
        self.env.forward(q=q_init, joint_names=self.joint_names, increase_tick=False)
        p_home, _ = self.env.get_pR_body(body_name="i10_inspire_flange_link")

        # Keep HOME-arm EE position but enforce downward flange orientation by IK.
        R_trgt = rpy2r(RESET_EE_RPY_RAD)
        q_zero, ik_err_stack, ik_info = solve_ik(
            env=self.env,
            joint_names_for_ik=self.joint_names,
            body_name_trgt="i10_inspire_flange_link",
            q_init=q_init,
            p_trgt=p_home,
            R_trgt=R_trgt,
        )
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)

        # Set object positions
        obj_names = self.env.get_body_names(prefix='cube')
        n_obj = len(obj_names)
        cx, cy, cz = float(CUBE_SPAWN_XYZ[0]), float(CUBE_SPAWN_XYZ[1]), float(CUBE_SPAWN_XYZ[2])
        obj_xyzs = sample_xyzs(
            n_obj,
            x_range=[cx - CUBE_SAMPLE_DX, cx + CUBE_SAMPLE_DX],
            y_range=[cy - CUBE_SAMPLE_DY, cy + CUBE_SAMPLE_DY],
            z_range=[cz - CUBE_SAMPLE_DZ, cz + CUBE_SAMPLE_DZ],
            min_dist=0.2,
            xy_margin=0.0,
        )
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx],p=obj_xyzs[obj_idx,:])
            self.env.set_R_base_body(body_name=obj_names[obj_idx],R=np.eye(3,3))
        self.env.forward(increase_tick=False)

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.array([0.0], dtype=np.float64)])
        self.p0, self.R0 = self.env.get_pR_body(body_name='i10_inspire_flange_link')
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        self.gripper_close = True
        self.past_chars = []

    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - qpos: [j1,j2,j3,j4,j5,j6]

        '''
        if self.action_type == 'ee_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'i10_inspire_flange_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_qpos':
            q = action[:-1] + self.last_q
        elif self.action_type == 'qpos':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')

        gripper_close = np.array([action[-1]], dtype=np.float64)
        self.compute_q = q
        q = np.concatenate([q, gripper_close])

        self.q = q
        if self.state_type == 'qpos':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_qpos' or self.action_type == 'delta_qpos':
            dq =  self.get_delta_q()
            # Keep delta_qpos as plain ndarray for now (no stable semantic tag needed yet).
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        # self.rgb_ego = self.env.get_fixed_cam_rgb(
        #     cam_name='egocentric')
        # # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        # #     cam_name='topview')
        # self.rgb_side = self.env.get_fixed_cam_rgb(
        #     cam_name='sideview')
        self.rgb_wrist=self.env.get_fixed_cam_rgb(
            cam_name='wrist_cam')
        return self.rgb_agent, self.rgb_wrist
        

    def render(self, teleop=False):
        '''
        Render the environment
        '''
        self.env.plot_time()
        # p_current, R_current = self.env.get_pR_body(body_name='i10_inspire_flange_link')
        # R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0]])
        # self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        # self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        # rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        # self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            # rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            rgb_wrist_view = add_title_to_img(self.rgb_wrist,text='Wrist View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_wrist_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_close = 1.0 if gripper[0] > 0.1 else 0.0
        q = np.concatenate([qpos, [gripper_close]], dtype=np.float32)
        # User-facing semantic name: qpos (even though internal config uses "qpos").
        return as_typed(q, type="qpos")
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Donward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([-0.002,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([0.002,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,0.002,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,-0.002,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.002])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.002])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_I):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_K):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_J):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_L):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_U):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_O):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_LEFT_BRACKET): # 左闭右开 1闭0开
            self.gripper_close =  True
            print("Gripper close")
        if self.env.is_key_pressed_once(key=glfw.KEY_RIGHT_BRACKET):
            self.gripper_close =  False
            print("Gripper open")
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_close],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_close = 1.0 if gripper[0] > 0.1 else 0.0
        gripper_delta=self.gripper_close - gripper_close
        return np.concatenate([delta, [gripper_delta]],dtype=np.float32)

    def check_success(self):
        '''
        True when the cube is on the place deck (aligned with mujoco_teleop_env place logic) and gripper is close.
        Tolerances come from geom place_target_deck sizes in myscene.xml.
        '''
        model = self.env.model
        deck_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "place_target_deck")
        if deck_gid < 0:
            raise RuntimeError("geom 'place_target_deck' not found (expected in myscene.xml)")
        gs = np.asarray(model.geom_size[deck_gid], dtype=np.float64)
        place_deck_half_z = float(gs[2])
        place_tol_xy = float(0.75 * min(gs[0], gs[1]))
        place_height_eps = float(0.25 * place_deck_half_z)

        p_cube = self.env.get_p_body("cube")
        p_platform = self.env.get_p_body("place_target_platform")
        xy_ok = np.linalg.norm(p_cube[:2] - p_platform[:2]) < place_tol_xy
        z_ok = p_cube[2] > p_platform[2] + place_deck_half_z - place_height_eps
        gripper_open = float(self.env.get_qpos_joint("rh_r1")[0]) < 2.8e-6
        #  close : 0.81454458 open :2.7e-6
        
        
        # Check if the end effector has moved up and away from the cube
        p_ee = self.env.get_p_body("i10_inspire_flange_link")
        ee_away = (p_ee[2] - p_cube[2]) > 0.20  # 10 cm above the cube
        # print(f"gripper_open: {gripper_open}")
        # print(f"gripper_qpos: {float(self.env.get_qpos_joint("rh_r1")[0])}")
        # print(f"gripper_state: {self.gripper_close}")
        # print(f"xy_ok: {xy_ok}")
        # print(f"z_ok: {z_ok}")
        # print(f"ee_away: {ee_away}")
        return bool(xy_ok and z_ok and gripper_open and ee_away)
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        p_mug = self.env.get_p_body('cube')
        p_plate = self.env.get_p_body('place_target_platform')
        return p_mug, p_plate
    
    def set_obj_pose(self, p_mug, p_plate):
        '''
        Set the object poses
        args:
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='cube',p=p_mug)
        self.env.set_R_base_body(body_name='cube',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='place_target_platform',p=p_plate)
        self.env.set_R_base_body(body_name='place_target_platform',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name='i10_inspire_flange_link')
        rpy = r2rpy(R) # note vla models 最常用，因为省token
        # quat= r2quat(R)
        ee = np.concatenate([p, rpy], dtype=np.float32)
        return as_typed(ee, type="ee_pose")

    def get_obs_action(self):
        '''
        get the observation and action
        returns:
            obs: np.array, observation
            action: np.array, action
        '''
        if self.action_type == 'ee_pose':
            obs_action = self.get_ee_pose()
        elif self.action_type == 'qpos':
            obs_action = self.get_joint_state()
        elif self.action_type == 'delta_qpos':
            obs_action = self.get_delta_q()
        else:
            raise ValueError('action_type not recognized')
        return obs_action