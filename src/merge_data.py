from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from src.features import add_temporal_features
except ModuleNotFoundError:
    from features import add_temporal_features


def merge_heart_rate_and_movement(project_dir: Path) -> pd.DataFrame:
    data_dir = project_dir / "data"
    hr_path = data_dir / "heart_rate.csv"
    movement_path = data_dir / "movement.csv"

    if not hr_path.exists():
        raise FileNotFoundError("Missing data/heart_rate.csv")
    if not movement_path.exists():
        raise FileNotFoundError("Missing data/movement.csv. Run src/video_features.py first.")

    hr = pd.read_csv(hr_path)
    movement = pd.read_csv(movement_path)

    required_hr = {"timestamp", "heart_rate", "state"}
    required_movement = {"timestamp", "movement", "state"}
    if not required_hr.issubset(hr.columns):
        raise ValueError(f"heart_rate.csv must include: {sorted(required_hr)}")
    if not required_movement.issubset(movement.columns):
        raise ValueError(f"movement.csv must include: {sorted(required_movement)}")

    hr["timestamp"] = hr["timestamp"].astype(int)
    movement["timestamp"] = movement["timestamp"].astype(int)

    merged = movement.merge(
        hr[["timestamp", "heart_rate", "state"]],
        on="timestamp",
        how="outer",
        suffixes=("_movement", "_hr"),
    ).sort_values("timestamp")

    merged["state"] = merged["state_movement"].combine_first(merged["state_hr"])
    merged = merged.drop(columns=["state_movement", "state_hr"])
    merged["movement"] = merged["movement"].interpolate().ffill().bfill()
    merged["heart_rate"] = merged["heart_rate"].interpolate().ffill().bfill()
    merged["state"] = merged["state"].ffill().bfill()
    merged = merged.dropna(subset=["movement", "heart_rate", "state"]).reset_index(drop=True)
    merged = add_temporal_features(merged)

    output_path = data_dir / "final_dataset.csv"
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize heart-rate and movement signals.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    df = merge_heart_rate_and_movement(args.project_dir)
    print(f"Saved {len(df)} synchronized rows to data/final_dataset.csv")


if __name__ == "__main__":
    main()

