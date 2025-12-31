# Live Speech Emotion Recognition (Web App) - Process & Flow

## Goal

Capture live microphone audio in a browser, send it to a backend service, and
return an emotion prediction using the trained LSTM model.

## System Flow (End-to-End)

1) Browser captures microphone audio.
2) Audio is encoded to WAV in the browser.
3) WAV file is sent to the FastAPI `/api/predict` endpoint.
4) Backend loads the audio, extracts MFCC features, and runs the LSTM model.
5) Backend returns emotion label + confidence + per-class scores.
6) UI renders the prediction and score breakdown.
   - Confidence doughnut chart
   - Per-emotion bar chart

Live streaming mode repeats steps 2-6 every few seconds while recording.

## Components

### Frontend (Browser)

- Uses `getUserMedia` to access the mic.
- Streams PCM frames using the Web Audio API.
- Encodes the PCM frames to a WAV file in JavaScript.
- Sends the WAV as `multipart/form-data` to the backend.

Key files:
- `app/static/index.html`
- `app/static/styles.css`
- `app/static/app.js`

### Backend (FastAPI)

- `POST /api/predict` accepts the WAV upload.
- Audio is loaded with `librosa`, then MFCC features are computed.
- The trained TensorFlow LSTM model predicts emotion.
- JSON response includes:
  - `label`
  - `confidence`
  - `scores` (all class probabilities)

Key files:
- `app/main.py`
- `app/model.py`
- `app/audio.py`

### Model Training

Training script uses the TESS dataset and matches the notebook pipeline:

1) Scan WAV files and infer labels from filenames.
2) Extract MFCCs (40 coefficients).
3) Expand dims to `(samples, 40, 1)`.
4) Train the LSTM model.
5) Save model to `models/ser_lstm.keras`.
6) Save label list to `models/labels.json`.

Key file:
- `train_model.py`
- `train_model_auto.py` (auto-detects dataset path)
- `evaluate_model.py` (generates plots and tables)

## Runbook (Local)

1) Activate venv and install dependencies.
2) Train model:
   - `python train_model.py --data-dir "<path to TESS dataset>"`
   - or `python train_model_auto.py --root "<project root>"`
3) Start server:
   - `uvicorn app.main:app --reload`
4) Open `http://127.0.0.1:8000` and allow mic access.

## Deployment Notes

- Deploy the FastAPI backend on Render or Railway.
- The model file `models/ser_lstm.keras` must exist on the server.
- For cloud deploys, train locally and commit the model file or mount it in
  the deployment artifact.

## Common Issues

- If the UI shows "Model not loaded", the model file is missing.
- If predictions are poor, retrain with more epochs or try data balancing.
- If audio upload fails, check mic permissions and browser console logs.
