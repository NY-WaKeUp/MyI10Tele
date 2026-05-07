import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeBuffer:
    episode_index: int
    task_index: int
    states: list[np.ndarray]
    actions: list[np.ndarray]
    timestamps: list[float]


class LeRobotV3Writer:
    """Write MuJoCo teleop trajectories with a LeRobot v3-like layout."""

    def __init__(
        self,
        dataset_root: str,
        repo_id: str,
        fps: int,
        robot_type: str,
        action_dim: int,
        state_dim: int,
        chunk_size: int = 1000,
    ) -> None:
        self.root = Path(dataset_root)
        self.repo_id = repo_id
        self.fps = fps
        self.robot_type = robot_type
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size

        self._episodes: list[dict] = []
        self._rows: list[dict] = []
        self._next_episode = 0

        self._prepare_dirs()

    def _prepare_dirs(self) -> None:
        (self.root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "videos" / "main" / "chunk-000").mkdir(parents=True, exist_ok=True)

    def new_episode_index(self) -> int:
        idx = self._next_episode
        self._next_episode += 1
        return idx

    def add_episode(self, episode: EpisodeBuffer, video_relpath: str) -> None:
        start_idx = len(self._rows)
        for frame_idx, (state, action, ts) in enumerate(
            zip(episode.states, episode.actions, episode.timestamps, strict=True)
        ):
            self._rows.append(
                {
                    "episode_index": episode.episode_index,
                    "frame_index": frame_idx,
                    "task_index": episode.task_index,
                    "timestamp": float(ts),
                    "observation.state": np.asarray(state, dtype=np.float32).tolist(),
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                }
            )

        end_idx = len(self._rows)
        self._episodes.append(
            {
                "episode_index": episode.episode_index,
                "task_index": episode.task_index,
                "length": len(episode.states),
                "from_row": start_idx,
                "to_row": end_idx,
                "video.main.path": video_relpath,
            }
        )

    def finalize(self, task_descriptions: list[str]) -> None:
        data_path = self.root / "data" / "chunk-000" / "file-000.parquet"
        episodes_path = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        tasks_path = self.root / "meta" / "tasks.parquet"
        info_path = self.root / "meta" / "info.json"
        stats_path = self.root / "meta" / "stats.json"

        pd.DataFrame(self._rows).to_parquet(data_path, index=False)
        pd.DataFrame(self._episodes).to_parquet(episodes_path, index=False)
        pd.DataFrame(
            [{"task_index": i, "task": task} for i, task in enumerate(task_descriptions)]
        ).to_parquet(tasks_path, index=False)

        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "fps": self.fps,
            "chunks_size": self.chunk_size,
            "data_path": "data/chunk-000/file-000.parquet",
            "video_path": "videos/main/chunk-000",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [self.state_dim]},
                "action": {"dtype": "float32", "shape": [self.action_dim]},
            },
            "repo_id": self.repo_id,
            "total_episodes": len(self._episodes),
            "total_frames": len(self._rows),
        }
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

        state_stack = np.asarray([row["observation.state"] for row in self._rows], dtype=np.float32)
        action_stack = np.asarray([row["action"] for row in self._rows], dtype=np.float32)
        stats = {
            "observation.state": {
                "mean": state_stack.mean(axis=0).tolist(),
                "std": state_stack.std(axis=0).tolist(),
                "min": state_stack.min(axis=0).tolist(),
                "max": state_stack.max(axis=0).tolist(),
            },
            "action": {
                "mean": action_stack.mean(axis=0).tolist(),
                "std": action_stack.std(axis=0).tolist(),
                "min": action_stack.min(axis=0).tolist(),
                "max": action_stack.max(axis=0).tolist(),
            },
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
