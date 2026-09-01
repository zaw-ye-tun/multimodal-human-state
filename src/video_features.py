from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


SESSION_ORDER = ["rest", "active", "recovery"]


def extract_movement(video_path: Path, state: str) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(round(fps)), 1)
    previous = None
    rows = []
    frame_count = 0
    sampled_second = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_count % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))

            if previous is None:
                movement = 0.0
            else:
                diff = cv2.absdiff(gray, previous)
                movement = float(diff.mean())

            rows.append(
                {
                    "timestamp": sampled_second,
                    "movement": movement,
                    "state": state,
                }
            )
            previous = gray
            sampled_second += 1

        frame_count += 1

    cap.release()
    return pd.DataFrame(rows)


def extract_all(project_dir: Path, sessions: list[str] | None = None) -> pd.DataFrame:
    sessions = sessions or SESSION_ORDER
    data_dir = project_dir / "data"
    video_dir = project_dir / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    offset = 0

    for state in sessions:
        video_path = video_dir / f"{state}.mp4"
        if not video_path.exists():
            print(f"Skipping missing video: {video_path}")
            continue

        local_df = extract_movement(video_path, state)
        local_df.to_csv(data_dir / f"{state}_movement.csv", index=False)

        global_df = local_df.copy()
        global_df["timestamp"] = global_df["timestamp"] + offset
        all_rows.append(global_df)

        if not local_df.empty:
            offset += int(local_df["timestamp"].max()) + 1

    if not all_rows:
        raise FileNotFoundError("No session videos found in videos/rest.mp4, active.mp4, recovery.mp4")

    movement = pd.concat(all_rows, ignore_index=True)
    movement.to_csv(data_dir / "movement.csv", index=False)
    return movement


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one-frame-per-second movement features.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--sessions", nargs="*", default=SESSION_ORDER)
    args = parser.parse_args()

    df = extract_all(args.project_dir, args.sessions)
    print(f"Saved {len(df)} movement rows to data/movement.csv")


if __name__ == "__main__":
    main()

