import numpy as np
from src.core.mujoco_teleop_env import MujocoTeleopEnv, StepOutput, default_model_path
from src.dataset.task_spec import TaskSpec


class MyEnv:
    def __init__(
        self,
        xml_path: str | None = None,
        action_type: str = "ee_pose",  # or 'qpos'
        state_type: str = "qpos",  # or 'ee_pose'
        seed: int = 0,
        ik_damping: float = 1e-3,
        ik_gain: float = 0.6,
    ) -> None:
        model_path = xml_path if xml_path is not None else default_model_path()
        self.backend = MujocoTeleopEnv(
            model_path=model_path,
            seed=seed,
            ik_damping=ik_damping,
            ik_gain=ik_gain,
        )
        self.action_type = action_type
        self.state_type = state_type
        self.default_task = TaskSpec(
            task_id="myenv_default",
            instruction="Move and manipulate cube with teleop action deltas.",
            cube_xy_low=np.array([0.20, -1.15], dtype=np.float64),
            cube_xy_high=np.array([0.33, -1.03], dtype=np.float64),
            max_steps=1000,
        )

    @property
    def model(self):
        return self.backend.model

    @property
    def data(self):
        return self.backend.data

    def reset(self, seed: int | None = None) -> np.ndarray:
        # Keep behavior deterministic when user specifies a seed.
        if seed is not None:
            self.backend.rng = np.random.default_rng(seed)
        self.backend.reset(self.default_task)
        return self.get_state()

    def step(self, action: np.ndarray, sim_steps_per_control: int = 1) -> StepOutput:
        return self.backend.step(
            action=np.asarray(action, dtype=np.float32),
            sim_steps_per_control=sim_steps_per_control,
            task=self.default_task,
        )

    def get_state(self) -> np.ndarray:
        return self.backend.get_state()

    def render_main(self, width: int = 640, height: int = 480) -> np.ndarray:
        return self.backend.render_main(width=width, height=height)

    def render_wrist(self, width: int = 640, height: int = 480) -> np.ndarray:
        return self.backend.render_wrist(width=width, height=height)