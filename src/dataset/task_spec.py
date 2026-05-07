from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    cube_xy_low: np.ndarray
    cube_xy_high: np.ndarray
    max_steps: int


TASK_SPECS = [
    TaskSpec(
        task_id="reach",
        instruction="Move the gripper center above the cube.",
        cube_xy_low=np.array([0.20, -1.15], dtype=np.float64),
        cube_xy_high=np.array([0.33, -1.03], dtype=np.float64),
        max_steps=600,
    ),
    TaskSpec(
        task_id="grasp",
        instruction="Grasp the cube securely with the gripper.",
        cube_xy_low=np.array([0.20, -1.15], dtype=np.float64),
        cube_xy_high=np.array([0.33, -1.03], dtype=np.float64),
        max_steps=800,
    ),
    TaskSpec(
        task_id="place",
        instruction="Place the cube on the target plate region.",
        cube_xy_low=np.array([0.20, -1.15], dtype=np.float64),
        cube_xy_high=np.array([0.33, -1.03], dtype=np.float64),
        max_steps=1000,
    ),
]
