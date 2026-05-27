#!/usr/bin/env python3
"""
Side-by-side comparison of a participant's marker qpos and video qpos
rendered trajectories.

Usage: python analysis/render_comparison.py <participant_number>
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import cv2
import numpy as np

from monocular_demos.biomechanics_mjx.visualize import render_trajectory

HERE = Path(__file__).parent
REPO = HERE.parent
DATA_DIR = REPO / "data" / "movi"
XML_PATH = REPO / "monocular_demos" / "biomechanics_mjx" / "data" / "humanoid" / "humanoid_torque.xml"


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side render: marker qpos (left) vs video qpos (right)"
    )
    parser.add_argument("participant", type=int, help="Participant number (e.g. 1)")
    parser.add_argument(
        "--output", "-o",
        help="Output video path (default: data/movi/participant_N_comparison.mp4)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--render-width", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames rendered")
    args = parser.parse_args()

    npz_path = DATA_DIR / f"participant_{args.participant}.npz"
    if not npz_path.exists():
        sys.exit(f"Data not found: {npz_path}")
    data = np.load(npz_path)
    marker_qpos = data["marker_qpos_euler"]   # (T, 40)
    video_qpos = data["video_qpos_aligned"]   # (T, 40)

    if args.max_frames is not None:
        marker_qpos = marker_qpos[:args.max_frames]
        video_qpos = video_qpos[:args.max_frames]

    print(f"Participant {args.participant}: {marker_qpos.shape[0]} frames")

    render_kwargs = dict(
        xml_path=str(XML_PATH),
        height=args.render_height,
        width=args.render_width,
        fps=int(args.fps),
    )

    print("Rendering marker qpos...")
    marker_frames = render_trajectory(pose=marker_qpos, **render_kwargs)

    print("Rendering video qpos...")
    video_frames = render_trajectory(pose=video_qpos, **render_kwargs)

    n = min(len(marker_frames), len(video_frames))
    total_w = args.render_width * 2

    output = args.output or str(DATA_DIR / f"participant_{args.participant}_comparison.mp4")
    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (total_w, args.render_height),
    )

    print("Writing output video...")
    for marker_frame, video_frame in zip(marker_frames[:n], video_frames[:n]):
        left = cv2.cvtColor(marker_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(video_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(np.concatenate([left, right], axis=1))

    writer.release()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
