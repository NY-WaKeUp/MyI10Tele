from lerobot.datasets.lerobot_dataset import LeRobotDataset
import shutil
import os
import numpy as np

ROOT = "./test_data"
if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

dataset = LeRobotDataset.create(
    repo_id="test",
    root=ROOT, 
    fps=20,
    features={
        "observation.image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["action"],
        },
    },
)

dataset.add_frame({
    "observation.image": np.zeros((256, 256, 3), dtype=np.uint8),
    "observation.state": np.zeros(7, dtype=np.float32),
    "action": np.zeros(6, dtype=np.float32),
    "task": "testing",
})
dataset.save_episode()
dataset.finalize()
print("Success!")
