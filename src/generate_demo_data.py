from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SESSIONS = [
    ("rest", 75, 4.0),
    ("active", 125, 34.0),
    ("recovery", 98, 14.0),
]


def make_demo_video(path: Path, state: str, seconds: int, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video file: {path}")

    rng = np.random.default_rng(42 + len(state))
    position = np.array([120.0, 90.0])
    velocity_by_state = {
        "rest": np.array([0.3, 0.2]),
        "active": np.array([3.5, 2.7]),
        "recovery": np.array([1.6, 1.1]),
    }
    velocity = velocity_by_state[state]

    for frame_index in range(seconds * fps):
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        noise = rng.normal(0, 2, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if state == "recovery":
            decay = 1.0 - (frame_index / max(seconds * fps - 1, 1)) * 0.55
            step = velocity * decay
        else:
            step = velocity

        position += step
        if position[0] < 20 or position[0] > 270:
            velocity[0] *= -1
        if position[1] < 20 or position[1] > 190:
            velocity[1] *= -1

        x, y = position.astype(int)
        cv2.rectangle(frame, (x, y), (x + 42, y + 42), (40, 95, 210), -1)
        cv2.circle(frame, (x + 70, y + 28), 18, (35, 150, 90), -1)
        cv2.putText(
            frame,
            state.upper(),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()


def make_demo_heart_rate(data_dir: Path, seconds_per_session: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    timestamp = 0

    for state, base_hr, _ in SESSIONS:
        for local_second in range(0, seconds_per_session, 15):
            if state == "rest":
                hr = base_hr + rng.normal(0, 1.5)
            elif state == "active":
                hr = base_hr + min(local_second / 4, 25) + rng.normal(0, 2.0)
            else:
                hr = base_hr - min(local_second / 5, 28) + rng.normal(0, 2.0)
            rows.append(
                {
                    "timestamp": timestamp + local_second,
                    "heart_rate": int(round(hr)),
                    "state": state,
                }
            )
        timestamp += seconds_per_session

    df = pd.DataFrame(rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "heart_rate.csv", index=False)
    return df


def generate_demo_data(project_dir: Path, seconds_per_session: int = 60, fps: int = 12) -> None:
    data_dir = project_dir / "data"
    video_dir = project_dir / "videos"

    for state, _, _ in SESSIONS:
        make_demo_video(video_dir / f"{state}.mp4", state, seconds_per_session, fps)
    make_demo_heart_rate(data_dir, seconds_per_session)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo videos and heart-rate CSV.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--seconds-per-session", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    generate_demo_data(args.project_dir, args.seconds_per_session, args.fps)
    print("Created demo videos in videos/ and data/heart_rate.csv")


if __name__ == "__main__":
    main()

