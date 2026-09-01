# Multimodal Human State Monitoring

This project explores whether visual behaviour and physiological signals can be combined to recognise basic human states in real-world conditions.

It is a one-day applied prototype, not a clinical system.

## Modalities

- Webcam-based movement from frame differences
- Heart rate from a CSV file
- Temporal/context features from short rolling windows

## Pipeline

Video + physiological data -> timestamp synchronisation -> feature extraction -> multimodal fusion -> machine-learning classification -> Streamlit dashboard.

## Architecture

```text
                 +-----------+
                 |  Webcam   |
                 +-----+-----+
                       |
                Movement features
                       |
                       v
+------------+   +-----+-----------+
| Heart rate |-->| Synchronisation |
+------------+   +-----+-----------+
                       |
                       v
              +--------+---------+
              | Multimodal       |
              | feature fusion   |
              +--------+---------+
                       |
                       v
              +--------+---------+
              | Machine learning |
              +--------+---------+
                       |
                       v
             Rest / Active / Recovery
```

## Research Question

Does combining visual and physiological information improve human-state recognition compared with individual modalities?

## Project Structure

```text
multimodal-human-state/
|-- data/
|-- videos/
|-- src/
|   |-- features.py
|   |-- generate_demo_data.py
|   |-- merge_data.py
|   |-- train_model.py
|   `-- video_features.py
|-- app.py
|-- requirements.txt
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Demo

The demo creates synthetic videos and heart-rate data so the full pipeline can run before you collect your own data.

```bash
python -m src.generate_demo_data
python -m src.video_features
python -m src.merge_data
python -m src.train_model
streamlit run app.py
```

## Collect Your Own Data

Record three short webcam sessions:

| Session | Activity | Suggested Length |
| --- | --- | --- |
| Rest | Sit quietly | 5 min |
| Active | Walk or light exercise | 5 min |
| Recovery | Sit after exercise | 5 min |

Save the videos as:

```text
videos/rest.mp4
videos/active.mp4
videos/recovery.mp4
```

Create `data/heart_rate.csv`:

```csv
timestamp,heart_rate,state
0,72,rest
15,71,rest
30,73,rest
300,118,active
315,122,active
600,91,recovery
615,88,recovery
```

Use cumulative relative seconds across the full experiment. Perfect timestamps are not required for this prototype.

Then run:

```bash
python -m src.video_features
python -m src.merge_data
python -m src.train_model
streamlit run app.py
```

## Outputs

- `data/rest_movement.csv`, `data/active_movement.csv`, `data/recovery_movement.csv`
- `data/movement.csv`
- `data/final_dataset.csv`
- `data/model_comparison.csv`
- `data/*_classification_report.txt`
- `human_state_model.pkl`

## Current Proof of Concept

Three states are evaluated:

- Rest
- Active
- Recovery

The model comparison tests:

- Heart rate only
- Camera movement only
- Heart rate + movement

Do not invent accuracy values. Use the actual values in `data/model_comparison.csv`.

## Scientific Limitation

This is a proof-of-concept dataset collected from one participant and is not intended to demonstrate generalisable clinical performance.

## Future Research

Future work may incorporate:

- Camera-based remote PPG
- ECG
- Radar sensing
- Body pose
- Facial/body behaviour
- Multimodal foundation models
- Contextual AI agents
- Larger multi-participant datasets
