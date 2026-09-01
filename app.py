from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import FEATURE_COLUMNS


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATASET_PATH = DATA_DIR / "final_dataset.csv"
MODEL_PATH = PROJECT_DIR / "human_state_model.pkl"
METRICS_PATH = DATA_DIR / "model_comparison.csv"


st.set_page_config(page_title="Multimodal Human State Monitoring", layout="wide")

st.title("Multimodal Human State Monitoring")
st.write(
    "Proof-of-concept combining webcam-derived movement and heart-rate data "
    "for Rest / Active / Recovery recognition."
)

if not DATASET_PATH.exists() or not MODEL_PATH.exists():
    st.warning("Dataset or model is missing.")
    st.write("Run these commands first:")
    st.code(
        "python -m src.generate_demo_data\n"
        "python -m src.video_features\n"
        "python -m src.merge_data\n"
        "python -m src.train_model",
        language="bash",
    )
    st.stop()

df = pd.read_csv(DATASET_PATH)
model = joblib.load(MODEL_PATH)
df["prediction"] = model.predict(df[FEATURE_COLUMNS])

latest = df.iloc[-1]
state_counts = df["prediction"].value_counts()

top = st.columns(4)
top[0].metric("Current predicted state", str(latest["prediction"]).title())
top[1].metric("Latest heart rate", f"{latest['heart_rate']:.0f} bpm")
top[2].metric("Latest movement", f"{latest['movement']:.2f}")
top[3].metric("Samples", f"{len(df):,}")

chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader("Heart-rate trend")
    st.line_chart(df.set_index("timestamp")["heart_rate"])

with chart_right:
    st.subheader("Visual movement")
    st.line_chart(df.set_index("timestamp")["movement"])

if METRICS_PATH.exists():
    st.subheader("Modality comparison")
    metrics = pd.read_csv(METRICS_PATH)
    display_metrics = metrics.copy()
    display_metrics["accuracy"] = (display_metrics["accuracy"] * 100).round(1)
    st.bar_chart(display_metrics.set_index("model")["accuracy"])
    st.dataframe(display_metrics, use_container_width=True)

st.subheader("Predicted human state over time")
st.dataframe(
    df[
        [
            "timestamp",
            "heart_rate",
            "movement",
            "state",
            "prediction",
        ]
    ],
    use_container_width=True,
)

st.subheader("Prediction distribution")
st.bar_chart(state_counts)

