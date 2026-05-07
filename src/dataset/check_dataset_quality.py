import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality checks for collected VLA dataset.")
    parser.add_argument("--dataset-root", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    data_path = root / "data" / "chunk-000" / "file-000000.parquet"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000000.parquet"
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"

    for p in [data_path, episodes_path, info_path, stats_path]:
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    df = pd.read_parquet(data_path)
    ep = pd.read_parquet(episodes_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))

    required_cols = {"episode_index", "frame_index", "task_index", "task", "timestamp", "observation.state", "action"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in data parquet: {sorted(missing)}")

    if df.isnull().any().any():
        raise ValueError("found NaN values in data parquet")

    if len(ep) == 0 or len(df) == 0:
        raise ValueError("empty dataset")

    success_rate = float(ep["success"].mean()) if "success" in ep.columns else 0.0
    frames_per_episode = float(len(df) / len(ep))
    task_counts = ep["task"].value_counts().to_dict()

    print("=== Dataset Quality Report ===")
    print(f"repo_id: {info.get('repo_id', 'N/A')}")
    print(f"total_episodes: {len(ep)}")
    print(f"total_frames: {len(df)}")
    print(f"avg_frames_per_episode: {frames_per_episode:.2f}")
    print(f"success_rate: {success_rate:.3f}")
    print(f"task_counts: {task_counts}")

    bad_videos = []
    for c in ["video.main.path", "video.wrist.path"]:
        if c in ep.columns:
            for rel in ep[c]:
                vp = root / rel
                if not vp.exists() or vp.stat().st_size == 0:
                    bad_videos.append(str(vp))
    if bad_videos:
        raise ValueError(f"missing or empty videos: {bad_videos[:10]}")

    print("dataset check passed")


if __name__ == "__main__":
    main()
