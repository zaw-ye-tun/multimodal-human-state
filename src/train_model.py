from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    from src.features import FEATURE_COLUMNS
except ModuleNotFoundError:
    from features import FEATURE_COLUMNS


MODEL_SPECS = {
    "heart_rate_only": ["heart_rate", "hr_change", "hr_rolling_mean"],
    "movement_only": ["movement", "movement_rolling_mean"],
    "multimodal": FEATURE_COLUMNS,
}


def train_and_compare(project_dir: Path) -> pd.DataFrame:
    data_dir = project_dir / "data"
    dataset_path = data_dir / "final_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError("Missing data/final_dataset.csv. Run src/merge_data.py first.")

    df = pd.read_csv(dataset_path).dropna(subset=FEATURE_COLUMNS + ["state"])
    if df["state"].nunique() < 2:
        raise ValueError("Need at least two states to train a classifier.")

    min_class_count = int(df["state"].value_counts().min())
    stratify = df["state"] if min_class_count >= 2 else None

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    metrics = []
    best_model = None
    best_accuracy = -1.0

    for name, columns in MODEL_SPECS.items():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(train_df[columns], train_df["state"])
        predictions = model.predict(test_df[columns])
        accuracy = accuracy_score(test_df["state"], predictions)
        metrics.append({"model": name, "features": ", ".join(columns), "accuracy": accuracy})

        report = classification_report(test_df["state"], predictions, zero_division=0)
        (data_dir / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")

        if name == "multimodal" or accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(data_dir / "model_comparison.csv", index=False)
    joblib.dump(best_model, project_dir / "human_state_model.pkl")
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest models and compare modalities.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    metrics = train_and_compare(args.project_dir)
    print(metrics.to_string(index=False))
    print("Saved model to human_state_model.pkl and metrics to data/model_comparison.csv")


if __name__ == "__main__":
    main()

