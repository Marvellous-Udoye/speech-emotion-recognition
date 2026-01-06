# LSTM-Based Speech Emotion Recognition for Student Public Speaking Training

This project builds an emotion-aware feedback system for student public speaking
training. It extracts MFCC features from speech audio, trains an LSTM-based
classifier to recognize seven emotions, and delivers live emotion feedback in a
web interface to support delivery stability and emotional control. Session data
is captured per device and aggregated across users to produce cohort-level
insights on confidence and emotional trends.

## Research scope

The system targets student public speaking practice sessions and provides
emotion cues (e.g., fear, sadness, anger, neutrality) as a proxy for delivery
stability and confidence trends during rehearsal, then aggregates those signals
across demographics to evaluate cohort-level patterns.

## Research objective

Design and evaluate an LSTM-based speech emotion recognition pipeline that
provides real-time emotion feedback for student public speaking training, with
quantitative evaluation of classification performance and qualitative analysis
of delivery stability cues, plus cohort analytics derived from aggregated
session data.

## Research contributions

- A reproducible MFCC + LSTM pipeline for speech emotion recognition on TESS.
- A live web application that captures microphone audio and displays emotion
  confidence scores for rehearsal feedback.
- An evaluation workflow that produces publication-ready metrics tables and
  visualizations (confusion matrix, ROC, PR curves, and accuracy/loss plots).
- A session analytics pipeline that aggregates per-device recordings into
  cohort summaries and confidence/emotion insights.

## Setup (Windows, Python 3.12)

Recommended: use a virtual environment to keep dependencies isolated.

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install ipykernel
```

If TensorFlow fails to import on Windows, install the Microsoft Visual C++
Redistributable 2015-2022 (x64), then restart the notebook kernel.

TensorFlow 2.16 on Windows requires `numpy<2.0`. This is pinned in
`requirements.txt`.

## Live demo (Web app)

Train the model (required once):

```bash
python train_model.py --data-dir "C:\Users\...\speech-emotion-recognition\TESS Toronto emotional speech set data"
```

Or auto-detect the dataset:

```bash
python train_model_auto.py --root "C:\Users\...\speech-emotion-recognition"
```

Run the web app:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` and allow microphone access. Turn on "Live
streaming" for continuous updates while recording.

## Notebook usage

1) Open `Speech Emotion Recognition - Sound Classification.ipynb`
2) Ensure the kernel is the project venv
3) Update the dataset path in the data-loading cell (see below)
4) Use TensorFlow Keras imports (`from tensorflow.keras ...`)
5) Run cells top to bottom

### Dataset location

The TESS dataset is downloaded and stored in the project root. The expected
folder name is `TESS Toronto emotional speech set data`.

## Expected results

The baseline LSTM model in the notebook typically reaches around 65-72%
validation accuracy depending on random seed and environment.

## Evaluation (figures + tables)

Generate paper-ready plots and metrics:

```bash
python evaluate_model.py --data-dir "C:\Users\...\speech-emotion-recognition\TESS Toronto emotional speech set data"
```

Artifacts are saved under `results/`.

## Session cohort analytics

Aggregate per-device session analytics and generate cohort insights:

```bash
python evaluate_sessions.py
```

Outputs are written to `results/aggregates/`:

- `session_summary.csv` / `session_summary.json` / `session_summary.md` for group totals
- `cohort_insights.md` / `cohort_insights.json` for human-readable cohort statements

## Session workflow (core project flow)

1) A user starts a session on their device and submits profile metadata with a
   `device_id`.
2) Each recording made during that session is stored with the `device_id` so
   per-session analytics can be computed later.
3) All session data across devices is aggregated to produce cohort summaries
   (e.g., confidence and top emotions by age group, gender, institution).

## Project structure

```
app/
  static/
    app.js
    index.html
    styles.css
models/
  history.json
  labels.json
  ser_lstm.keras
results/
  aggregates/
    cohort_insights.json
    cohort_insights.md
    session_summary.csv
    session_summary.json
    session_summary.md
  users/
    <device_id>/
      predictions.jsonl
      session.json
train_model.py
train_model_auto.py
evaluate_model.py
evaluate_sessions.py
```

## Example data (aggregates + users)

`results/aggregates/session_summary.json`:

```json
{
  "totals": {
    "devices": 1,
    "recordings": 10,
    "avg_confidence": 0.9999,
    "emotion_counts": {
      "disgust": 10
    }
  },
  "groups": [
    {
      "gender": "Male",
      "age_group": "16-17",
      "institution": "University",
      "level": "100 Level",
      "faculty": "Engineering",
      "presentation_type": "Class presentation",
      "experience": "Beginner",
      "devices": 1,
      "recordings": 10,
      "avg_confidence": 0.9999,
      "top_emotion": "disgust"
    }
  ]
}
```

`results/aggregates/session_summary.csv`:

```csv
gender,age_group,institution,level,faculty,presentation_type,experience,devices,recordings,avg_confidence,top_emotion
Male,16-17,University,100 Level,Engineering,Class presentation,Beginner,1,10,0.9999,disgust
```

`results/aggregates/session_summary.md`:

```md
# Session Analytics Summary

- Devices: 1
- Recordings: 10
- Average confidence: 0.9999

## Top emotions (overall)
- disgust: 10
```

`results/aggregates/cohort_insights.md`:

```md
# Cohort Insights

- Male University students age 16-17 show very high confidence (avg 0.9999) with top emotion disgust.
```

`results/aggregates/cohort_insights.json`:

```json
[
  {
    "cohort": "Male University students age 16-17",
    "confidence_level": "very high",
    "avg_confidence": 0.9999,
    "top_emotion": "disgust",
    "recordings": 10
  }
]
```

`results/users/<device_id>/session.json`:

```json
{
  "device_id": "1866670e-932f-4531-acfe-4f33e8106fbb",
  "profile": {
    "age": "16-17",
    "gender": "Male",
    "institution": "University",
    "level": "100 Level",
    "faculty": "Engineering",
    "presentation": "Class presentation",
    "experience": "Beginner"
  },
  "created_at": "2026-01-06T19:56:23.537Z",
  "updated_at": "2026-01-06T19:56:23.615460+00:00",
  "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
}
```

`results/users/<device_id>/predictions.jsonl` (one JSON object per line):

```json
{"timestamp":"2026-01-06T19:56:29.688896+00:00","device_id":"1866670e-932f-4531-acfe-4f33e8106fbb","label":"disgust","confidence":0.9998071789741516,"scores":{"angry":2.636892986629391e-06,"disgust":0.9998071789741516,"fear":1.4559319971851892e-08,"happy":8.728329703444615e-06,"neutral":1.742362520928964e-08,"ps":0.00015552714467048645,"sad":2.593634599179495e-05},"audio_bytes_len":96044,"content_type":"audio/wav","upload_name":"recording.wav","upload_source":"live"}
```

## Dataset Information

There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.
The dataset is organised such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format.

## Output Attributes

* anger
* disgust
* fear
* happiness
* pleasant surprise
* sadness
* neutral

## Download Links

Download link: [https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)
More Datasets: [https://www.kaggle.com/dmitrybabko/speech-emotion-recognition-en](https://www.kaggle.com/dmitrybabko/speech-emotion-recognition-en)

## Libraries

* pandas
* matplotlib
* tensorflow
* librosa

## Neural Network

LSTM Network
Validation accuracy: ~65-72% (baseline)
