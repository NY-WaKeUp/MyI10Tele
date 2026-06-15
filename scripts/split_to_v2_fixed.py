#!/usr/bin/env python3
"""Convert LeRobot v3.0 chunked dataset to v2.0 per-episode layout for openpi.

- Parquet: groupby episode_index -> data/chunk-000/episode_XXXXXX.parquet
- Video: cut using v3 meta/episodes offsets (file_index + from_timestamp), NOT global concat
- Meta: copy episodes.jsonl / tasks / stats, patch info.json to v2 paths
- Optional --to-v21: per-episode stats + codebase_version v2.1 (local only, no Hub push)

Requires lerobot (e.g. openpi venv): uv run python .../split_to_v2_fixed.py --to-v21 ...
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ChunkInfo:
    path: Path
    num_frames: int
    codec: str


@dataclass
class EpisodeRow:
    episode_index: int
    length: int
    tasks: list[str]
    video_file_index: dict[str, int]
    video_from_timestamp: dict[str, float]


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ffprobe_frame_count(path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()
    if not out.isdigit():
        raise RuntimeError(f"Could not count frames in {path}: {out!r}")
    return int(out)


def ffprobe_codec(path: Path) -> str:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()
    return out


def load_episode_meta(src: Path, cameras: list[str]) -> list[EpisodeRow]:
    ep_dir = src / "meta" / "episodes"
    if not ep_dir.exists():
        raise FileNotFoundError(f"Missing v3 episode metadata under {ep_dir}")
    ep_df = pd.concat(
        [pd.read_parquet(p) for p in sorted(ep_dir.rglob("*.parquet"))],
        ignore_index=True,
    ).sort_values("episode_index")

    rows: list[EpisodeRow] = []
    for _, r in ep_df.iterrows():
        ep_idx = int(r["episode_index"])
        tasks = r["tasks"]
        if isinstance(tasks, str):
            tasks = [tasks]
        rows.append(
            EpisodeRow(
                episode_index=ep_idx,
                length=int(r["length"]),
                tasks=list(tasks),
                video_file_index={
                    cam: int(r[f"videos/{cam}/file_index"]) for cam in cameras
                },
                video_from_timestamp={
                    cam: float(r[f"videos/{cam}/from_timestamp"]) for cam in cameras
                },
            )
        )
    return rows


def load_chunks(chunk_dir: Path) -> list[ChunkInfo]:
    paths = sorted(chunk_dir.glob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No chunk mp4 files under {chunk_dir}")
    chunks: list[ChunkInfo] = []
    for path in paths:
        n = ffprobe_frame_count(path)
        codec = ffprobe_codec(path)
        chunks.append(ChunkInfo(path=path, num_frames=n, codec=codec))
        print(f"    {path.name}: {n} frames ({codec})")
    return chunks


def plan_episode_segment(
    chunks: list[ChunkInfo],
    file_index: int,
    from_timestamp: float,
    length: int,
    fps: float,
) -> tuple[ChunkInfo, int, int]:
    start_frame = round(from_timestamp * fps)
    chunk = chunks[file_index]
    if start_frame + length > chunk.num_frames:
        raise RuntimeError(
            f"{chunk.path.name}: need frames [{start_frame}:{start_frame + length}) "
            f"but file has {chunk.num_frames}"
        )
    return chunk, start_frame, length


def extract_segment_h264(
    chunk: ChunkInfo,
    local_start: int,
    num_frames: int,
    out_path: Path,
    fps: float,
) -> None:
    local_end = local_start + num_frames - 1
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(chunk.path),
            "-vf",
            f"select=between(n\\,{local_start}\\,{local_end}),setpts=N/{fps}/TB",
            "-vsync",
            "cfr",
            "-frames:v",
            str(num_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            str(out_path),
        ]
    )
    got = ffprobe_frame_count(out_path)
    if got != num_frames:
        raise RuntimeError(f"{out_path}: expected {num_frames} frames, got {got}")


def split_parquet(
    src: Path,
    dst: Path,
    episode_rows: list[EpisodeRow],
    episode_filter: set[int] | None,
) -> None:
    data_chunk = dst / "data" / "chunk-000"
    data_chunk.mkdir(parents=True, exist_ok=True)

    src_chunks = sorted((src / "data" / "chunk-000").glob("*.parquet"))
    if not src_chunks:
        raise FileNotFoundError(f"No parquet under {src / 'data' / 'chunk-000'}")

    print("\n=== Parquet ===")
    df = pd.concat([pd.read_parquet(p) for p in src_chunks], ignore_index=True)

    for row in episode_rows:
        ep_idx = row.episode_index
        if episode_filter is not None and ep_idx not in episode_filter:
            continue
        ep_df = df[df["episode_index"] == ep_idx]
        if len(ep_df) != row.length:
            raise RuntimeError(
                f"episode {ep_idx}: parquet rows {len(ep_df)} != meta length {row.length}"
            )
        out_path = data_chunk / f"episode_{ep_idx:06d}.parquet"
        ep_df.to_parquet(out_path, index=False)
        print(f"  episode {ep_idx}: {len(ep_df)} rows -> {out_path.name}")


def split_camera(
    src_videos: Path,
    dst_videos: Path,
    cam: str,
    episode_rows: list[EpisodeRow],
    fps: float,
    episode_filter: set[int] | None,
) -> None:
    chunk_dir = src_videos / cam / "chunk-000"
    out_dir = dst_videos / "chunk-000" / cam
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Video: {cam} ===")
    print("  source chunks:")
    chunks = load_chunks(chunk_dir)

    for row in episode_rows:
        ep_idx = row.episode_index
        length = row.length
        if episode_filter is not None and ep_idx not in episode_filter:
            continue

        file_index = row.video_file_index[cam]
        from_ts = row.video_from_timestamp[cam]
        chunk, local_start, n = plan_episode_segment(
            chunks, file_index, from_ts, length, fps
        )

        out_mp4 = out_dir / f"episode_{ep_idx:06d}.mp4"
        print(
            f"\n  episode {ep_idx}: {n} frames from {chunk.path.name}"
            f"[{local_start}:{local_start + n}) (file_index={file_index}, from_ts={from_ts:.2f}s)"
        )

        extract_segment_h264(chunk, local_start, n, out_mp4, fps)

        got = ffprobe_frame_count(out_mp4)
        if got != length:
            raise RuntimeError(f"{out_mp4}: wrote {got} frames, expected {length}")
        dur = float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(out_mp4),
                ],
                text=True,
            ).strip()
        )
        expected_dur = length / fps
        if abs(dur - expected_dur) > 0.25:
            raise RuntimeError(
                f"{out_mp4}: duration {dur:.3f}s != expected {expected_dur:.3f}s (bad PTS?)"
            )
        print(f"    -> {out_mp4.name} ({got} frames, {dur:.2f}s)")


def setup_meta(src: Path, dst: Path, fps: float) -> None:
    dst_meta = dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    for name in ("episodes.jsonl", "tasks.jsonl", "stats.json"):
        src_file = src / "meta" / name
        if src_file.exists():
            shutil.copy2(src_file, dst_meta / name)

    info_path = dst_meta / "info.json"
    with open(src / "meta" / "info.json", encoding="utf-8") as f:
        info = json.load(f)

    info["codebase_version"] = "v2.0"
    info["fps"] = int(fps)
    info["data_path"] = (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    info["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    for key in list(info.get("features", {})):
        feat = info["features"][key]
        if feat.get("dtype") == "video" and "info" in feat:
            feat["info"]["video.codec"] = "h264"
            feat["info"]["video.pix_fmt"] = "yuv420p"
            feat["info"]["video.fps"] = int(fps)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {info_path} (codebase_version=v2.0)")


def verify_dataset(
    dst: Path, episode_rows: list[EpisodeRow], cameras: list[str], fps: float
) -> None:
    print("\n=== Verify ===")
    data_dir = dst / "data" / "chunk-000"
    for row in episode_rows:
        ep_idx = row.episode_index
        pq = data_dir / f"episode_{ep_idx:06d}.parquet"
        if not pq.exists():
            raise RuntimeError(f"Missing {pq}")
        n = len(pd.read_parquet(pq))
        if n != row.length:
            raise RuntimeError(f"{pq.name}: {n} rows != length {row.length}")

    for cam in cameras:
        vid_dir = dst / "videos" / "chunk-000" / cam
        for row in episode_rows:
            ep_idx = row.episode_index
            mp4 = vid_dir / f"episode_{ep_idx:06d}.mp4"
            if not mp4.exists():
                raise RuntimeError(f"Missing {mp4}")
            got = ffprobe_frame_count(mp4)
            if got != row.length:
                raise RuntimeError(f"{mp4.name}: {got} frames != {row.length}")

    print(f"  OK: {len(episode_rows)} episodes, parquet + {len(cameras)} camera(s)")


def write_tasks_jsonl(meta_dir: Path, task_override: str | None = None) -> int:
    """Write meta/tasks.jsonl from tasks.parquet or info.json fallback."""
    tasks_parquet = meta_dir / "tasks.parquet"
    if tasks_parquet.exists():
        tasks_df = pd.read_parquet(tasks_parquet)
        with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
            for task, row in tasks_df.iterrows():
                f.write(
                    json.dumps(
                        {"task_index": int(row["task_index"]), "task": str(task)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(tasks_df)
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    task_name = task_override or str(
        info.get("task") or info.get("default_task") or "Put cube on the black platform"
    )
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": task_name}) + "\n")
    return 1


def default_task_name(meta_dir: Path) -> str:
    tasks_jsonl = meta_dir / "tasks.jsonl"
    if tasks_jsonl.exists():
        first = json.loads(tasks_jsonl.read_text(encoding="utf-8").splitlines()[0])
        return str(first["task"])
    return "default_task"


def write_episodes_jsonl_from_data_parquets(dataset_root: Path, meta_dir: Path) -> int:
    """LeRobot v2.0: only per-episode data parquet, no meta/episodes/*.parquet."""
    task_name = default_task_name(meta_dir)
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"No data/ under {dataset_root}")
    episode_paths = sorted(data_root.rglob("episode_*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.parquet under {data_root}")

    records: list[dict] = []
    for ep_path in episode_paths:
        stem = ep_path.stem
        episode_index = int(stem.split("_", 1)[1])
        df = pd.read_parquet(ep_path, columns=["episode_index"])
        length = int(len(df))
        records.append(
            {
                "episode_index": episode_index,
                "tasks": [task_name],
                "length": length,
            }
        )
    records.sort(key=lambda r: r["episode_index"])
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def write_episodes_jsonl(dataset_root: Path, meta_dir: Path) -> int:
    ep_dir = meta_dir / "episodes"
    if ep_dir.is_dir():
        dfs = [pd.read_parquet(p) for p in sorted(ep_dir.rglob("*.parquet"))]
        ep = pd.concat(dfs, ignore_index=True)
    elif (meta_dir / "episodes.parquet").exists():
        ep = pd.read_parquet(meta_dir / "episodes.parquet")
    else:
        return write_episodes_jsonl_from_data_parquets(dataset_root, meta_dir)

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for _, row in ep.sort_values("episode_index").iterrows():
            tasks = row["tasks"]
            if hasattr(tasks, "tolist"):
                tasks = tasks.tolist()
            rec = {
                "episode_index": int(row["episode_index"]),
                "tasks": tasks,
                "length": int(row["length"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(ep)


def ensure_meta_jsonl(dataset_root: Path, task: str | None = None) -> None:
    """Ensure meta/tasks.jsonl and meta/episodes.jsonl exist (openpi + v2.1 convert)."""
    meta = dataset_root / "meta"
    if (meta / "episodes.jsonl").is_file() and (meta / "tasks.jsonl").is_file():
        return
    if not (meta / "info.json").exists():
        raise FileNotFoundError(f"Not a LeRobot dataset: {dataset_root}")
    print("\n=== Meta jsonl ===")
    n_tasks = write_tasks_jsonl(meta, task_override=task)
    n_ep = write_episodes_jsonl(dataset_root, meta)
    print(f"  Wrote {meta / 'tasks.jsonl'} ({n_tasks} tasks)")
    print(f"  Wrote {meta / 'episodes.jsonl'} ({n_ep} episodes)")


def convert_to_v21(
    dataset_root: Path,
    repo_id: str,
    num_workers: int,
    *,
    verify_stats: bool = False,
) -> None:
    """v2.0 -> v2.1 on local disk: episodes_stats.jsonl, drop global stats.json, no Hub."""
    print("\n=== LeRobot v2.0 -> v2.1 (local) ===")
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.common.datasets.utils import (
        EPISODES_STATS_PATH,
        STATS_PATH,
        load_stats,
        write_info,
    )
    from lerobot.common.datasets.v21.convert_stats import (
        check_aggregate_stats,
        convert_stats,
    )

    ds = LeRobotDataset(
        repo_id,
        root=dataset_root,
        revision="v2.0",
        force_cache_sync=False,
    )
    ep_stats_path = dataset_root / EPISODES_STATS_PATH
    if ep_stats_path.is_file():
        ep_stats_path.unlink()

    convert_stats(ds, num_workers=num_workers)

    stats_path = dataset_root / STATS_PATH
    if verify_stats and stats_path.is_file():
        ref_stats = load_stats(dataset_root)
        check_aggregate_stats(ds, ref_stats)
    elif stats_path.is_file():
        print(
            "  Skipping check_aggregate_stats: copied v3 stats.json often has wrong "
            "video/image std; per-episode stats from convert_stats are authoritative. "
            "Pass --verify-v21-stats to enforce the upstream check."
        )

    ds.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(ds.meta.info, dataset_root)

    if stats_path.is_file():
        stats_path.unlink()

    print(f"  {dataset_root}: codebase_version={CODEBASE_VERSION}")


def main(
    src_root: str = os.path.expanduser(
        "~/MyI10Tele/data_auboI10_ee_pose_v30_continuous_correctobjinit_gripperkp3000force100"
    ),
    dst_root: str = os.path.expanduser(
        "~/MyI10Tele/data_auboI10_ee_pose_v21_continuous_correctobjinit_gripperkp3000force100"
    ),
    fps: float = 20.0,
    cameras: list[str] | None = None,
    episodes: list[int] | None = None,
    skip_parquet: bool = False,
    skip_videos: bool = False,
    skip_meta: bool = False,
    clean: bool = False,
    to_v21: bool = True,
    v21_only: bool = False,
    verify_v21_stats: bool = False,
    repo_id: str = "auboI10",
    task: str | None = None,
    v21_num_workers: int = max(1, os.cpu_count()),
) -> None:
    """Convert v3 chunked dataset to v2 per-episode layout (openpi-compatible).

    ``to_v21``: after v2.0 split, upgrade ``dst_root`` to LeRobot v2.1 (local, no Hub).
    ``v21_only``: only run v2.0->v2.1 on existing ``dst_root`` (skip v3 split).
    ``verify_v21_stats``: run lerobot's check_aggregate_stats against meta/stats.json
    (usually fails when stats were copied from v3.0 — video std differs from recomputed).
    Use openpi venv (``lerobot`` installed). Safe with ``HF_HUB_OFFLINE=1``.
    """
    dst = Path(dst_root)
    if v21_only:
        if not (dst / "meta" / "info.json").exists():
            raise FileNotFoundError(f"Missing {dst / 'meta' / 'info.json'}")
        ensure_meta_jsonl(dst, task=task)
        convert_to_v21(
            dst,
            repo_id=repo_id,
            num_workers=v21_num_workers,
            verify_stats=verify_v21_stats,
        )
        print("\nDone (v2.1 only):", dst)
        return

    src = Path(src_root)
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing {src / 'meta' / 'info.json'}")

    src_videos = src / "videos"
    cam_names = cameras or sorted(p.name for p in src_videos.iterdir() if p.is_dir())
    episode_rows = load_episode_meta(src, cam_names)
    ep_filter = set(episodes) if episodes is not None else None

    print(f"Source: {src}")
    print(f"Dest:   {dst}")
    print(
        f"Episodes: {len(episode_rows)}"
        + (f", filter {sorted(ep_filter)}" if ep_filter else "")
    )

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")

    if clean and dst.exists():
        print(f"Removing {dst}")
        shutil.rmtree(dst)

    if not skip_meta:
        setup_meta(src, dst, fps)

    if not skip_parquet:
        split_parquet(src, dst, episode_rows, ep_filter)

    if not skip_videos:
        for cam in cam_names:
            split_camera(
                src / "videos", dst / "videos", cam, episode_rows, fps, ep_filter
            )

    rows_to_verify = [
        r for r in episode_rows if ep_filter is None or r.episode_index in ep_filter
    ]
    if ep_filter is None:
        rows_to_verify = episode_rows
    verify_dataset(dst, rows_to_verify, cam_names, fps)

    if to_v21:
        ensure_meta_jsonl(dst, task=task)
        convert_to_v21(
            dst,
            repo_id=repo_id,
            num_workers=v21_num_workers,
            verify_stats=verify_v21_stats,
        )

    print("\nDone. Point openpi --data.lerobot-root to:", dst)
    if to_v21:
        print("(LeRobot codebase_version v2.1 — training v2.0 warning should be gone)")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
