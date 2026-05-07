import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task_index: int
    task: str
    state: list[np.ndarray]
    action: list[np.ndarray]
    timestamp: list[float]
    success: bool
    video_main_relpath: str
    video_wrist_relpath: str


class LeRobotV3DatasetWriter:
    def __init__(self, root: str, fps: int, repo_id: str) -> None:
        self.root = Path(root)
        self.fps = fps
        self.repo_id = repo_id
        self.rows: list[dict] = []
        self.episodes: list[dict] = []
        self.task_to_index: dict[str, int] = {}
        self._prepare_dirs()

    def _prepare_dirs(self) -> None:
        for p in [
            self.root / "data" / "chunk-000",
            self.root / "meta" / "episodes" / "chunk-000",
            self.root / "videos" / "main" / "chunk-000",
            self.root / "videos" / "wrist" / "chunk-000",
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def episode_video_paths(self, episode_index: int) -> tuple[str, str]:
        return (
            f"videos/main/chunk-000/file-{episode_index:06d}.mp4",
            f"videos/wrist/chunk-000/file-{episode_index:06d}.mp4",
        )

    def add_episode(self, ep: EpisodeRecord) -> None:
        if ep.task not in self.task_to_index:
            self.task_to_index[ep.task] = len(self.task_to_index)
        task_index = self.task_to_index[ep.task]
        start = len(self.rows)
        for i, (s, a, t) in enumerate(zip(ep.state, ep.action, ep.timestamp, strict=True)):
            self.rows.append(
                {
                    "episode_index": ep.episode_index,
                    "frame_index": i,
                    "task_index": task_index,
                    "task": ep.task,
                    "timestamp": float(t),
                    "observation.state": np.asarray(s, dtype=np.float32).tolist(),
                    "action": np.asarray(a, dtype=np.float32).tolist(),
                }
            )
        end = len(self.rows)
        self.episodes.append(
            {
                "episode_index": ep.episode_index,
                "task_index": task_index,
                "task": ep.task,
                "length": len(ep.state),
                "from_row": start,
                "to_row": end,
                "success": ep.success,
                "video.main.path": ep.video_main_relpath,
                "video.wrist.path": ep.video_wrist_relpath,
            }
        )

    def finalize(self) -> None:
        pd.DataFrame(self.rows).to_parquet(self.root / "data" / "chunk-000" / "file-000000.parquet", index=False)
        pd.DataFrame(self.episodes).to_parquet(
            self.root / "meta" / "episodes" / "chunk-000" / "file-000000.parquet", index=False
        )
        with (self.root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
            for task, idx in sorted(self.task_to_index.items(), key=lambda kv: kv[1]):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

        state = np.asarray([r["observation.state"] for r in self.rows], dtype=np.float32)
        action = np.asarray([r["action"] for r in self.rows], dtype=np.float32)
        (self.root / "meta" / "stats.json").write_text(
            json.dumps(
                {
                    "observation.state": {
                        "mean": state.mean(0).tolist(),
                        "std": state.std(0).tolist(),
                        "min": state.min(0).tolist(),
                        "max": state.max(0).tolist(),
                    },
                    "action": {
                        "mean": action.mean(0).tolist(),
                        "std": action.std(0).tolist(),
                        "min": action.min(0).tolist(),
                        "max": action.max(0).tolist(),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.root / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "repo_id": self.repo_id,
                    "fps": self.fps,
                    "features": {
                        "observation.state": {"dtype": "float32", "shape": [7]},
                        "action": {"dtype": "float32", "shape": [7]},
                        "observation.images.main": {"dtype": "video"},
                        "observation.images.wrist": {"dtype": "video"},
                    },
                    "total_episodes": len(self.episodes),
                    "total_frames": len(self.rows),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
