#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal episode video recorder for MuJoCo environments."""

from pathlib import Path
from typing import Optional

import cv2
import imageio
import numpy as np


class EpisodeVideoRecorder:
    """Record episode videos with success/failure labeling.

    This utility records side-by-side agent and wrist camera views during
    policy evaluation episodes. Videos are automatically labeled with their
    outcome (success/failure) in the filename.

    Example usage:
        recorder = EpisodeVideoRecorder(output_dir="./videos", fps=20)

        for episode in range(num_episodes):
            recorder.start_episode(episode)

            while not done:
                agent_img, wrist_img = env.grab_image()
                recorder.record_frame(agent_img, wrist_img)
                # ... run policy ...

            recorder.stop(success=episode_success)
    """

    def __init__(
        self,
        output_dir: str = "./episode_videos",
        fps: int = 20,
        frame_size: tuple = (512, 256),  # (width, height) for side-by-side views
    ):
        """Initialize the video recorder.

        Args:
            output_dir: Directory to save recorded videos
            fps: Frames per second for the output video
            frame_size: Output video dimensions (width, height). The agent and
                       wrist views will be placed side-by-side, each taking
                       half the width.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self._frames: list[np.ndarray] = []
        self._pending_path: Optional[Path] = None
        self._current_episode: int = 0
        self._is_recording: bool = False

    def start_episode(self, episode_idx: int):
        """Start recording a new episode.

        Args:
            episode_idx: Episode index for filename generation
        """
        self.stop()
        self._current_episode = episode_idx
        timestamp = self._get_timestamp()
        self._pending_path = (
            self.output_dir / f"episode_{episode_idx:04d}_{timestamp}.mp4"
        )
        self._frames = []
        self._is_recording = True

    def record_frame(self, agent_img: np.ndarray, wrist_img: np.ndarray):
        """Record a single frame from agent and wrist cameras.

        Args:
            agent_img: Agent view image (H, W, 3) in RGB format
            wrist_img: Wrist view image (H, W, 3) in RGB format
        """
        if not self._is_recording:
            return

        half_width = self.frame_size[0] // 2
        half_height = self.frame_size[1]
        label_color_bgr = (0, 255, 0)

        combined = np.hstack(
            [
                self._annotate_view(
                    agent_img, "Agent View", half_width, half_height, label_color_bgr
                ),
                self._annotate_view(
                    wrist_img, "Wrist View", half_width, half_height, label_color_bgr
                ),
            ]
        )
        self._frames.append(combined)

    def _annotate_view(
        self,
        rgb_img: np.ndarray,
        label: str,
        width: int,
        height: int,
        color_bgr: tuple[int, int, int],
    ) -> np.ndarray:
        """Resize in BGR for OpenCV, draw label, return RGB for imageio/ffmpeg."""
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, (width, height))
        cv2.putText(
            resized,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_bgr,
            2,
            cv2.LINE_AA,
        )
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def stop(self, success: Optional[bool] = None):
        """Stop recording and optionally rename file with outcome.

        Args:
            success: If provided, renames the video file to include
                    '_success' or '_failure' suffix
        """
        saved_path: Optional[Path] = None
        if self._is_recording and self._frames and self._pending_path is not None:
            # H.264 in MP4: plays in VS Code, browsers, and ffplay. OpenCV mp4v is MPEG-4
            # Part 2 and is rejected by most modern players.
            imageio.mimsave(
                self._pending_path,
                self._frames,
                fps=self.fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
            )
            saved_path = self._pending_path

        self._frames = []
        self._pending_path = None

        if saved_path is not None and success is not None:
            outcome = "success" if success else "failure"
            new_path = saved_path.with_name(
                saved_path.name.replace(".mp4", f"_{outcome}.mp4")
            )
            saved_path.rename(new_path)

        self._is_recording = False

    def _get_timestamp(self) -> str:
        """Get current timestamp string for filename."""
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def __del__(self):
        """Cleanup on destruction."""
        self.stop()
