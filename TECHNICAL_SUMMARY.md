# Technical Summary

## What was done so far

- Created `requirements.txt` with TensorFlow 2.16.1 and `numpy<2.0` pins for
  Windows compatibility.
- Added a FastAPI backend with a model loader and `/api/predict` endpoint for
  emotion inference.
- Added a modern frontend that captures microphone audio, encodes WAV, and
  renders emotion predictions.
- Implemented live streaming mode (continuous updates while recording).
- Added model training scripts (`train_model.py` and `train_model_auto.py`) and
  a reusable training function.
- Added deployment configs for Render and Railway.
- Documented the end-to-end flow in `LIVE_APP_FLOW.md`.

## Notebook cell guide (by cell number)

Cell 1: Import core libraries (pandas, numpy, librosa, matplotlib, etc.) and
silence warnings.

Cell 3: Walk the dataset directory, collect audio file paths and labels based
on file naming, stop when 2800 files are found.

Cell 4: Check how many files were collected.

Cell 5: Preview a few file paths.

Cell 6: Preview a few labels.

Cell 7: Create the main DataFrame with `speech` (path) and `label`.

Cell 8: Count label distribution.

Cell 10: Visualize label distribution as a count plot.

Cell 11: Define helper functions to plot waveforms and spectrograms.

Cells 12-18: For each emotion, load one example audio file, plot waveform and
spectrogram, and play the audio.

Cell 20: Define MFCC extraction (mean of 40 coefficients over a 3s slice).

Cell 21: Test MFCC extraction on the first audio file.

Cell 22: Apply MFCC extraction to every file path in the DataFrame.

Cell 23: Inspect the extracted MFCC series.

Cell 24: Convert MFCC series to a NumPy array and check shape.

Cell 25: Expand dimensions to match LSTM input shape `(n_samples, 40, 1)`.

Cell 26: One-hot encode labels.

Cell 27: Convert labels to a dense NumPy array.

Cell 28: Check label array shape.

Cell 30: Build and compile the LSTM model (stacked Dense + Dropout).

Cell 31: Train the model with a validation split.

Cell 32: Notes from the tutorial about best validation accuracy and next steps.

Cell 34: Plot train/validation accuracy over epochs.

Cell 35: Plot train/validation loss over epochs.

## Web app architecture

### Frontend (browser)

Files:
- `app/static/index.html`
- `app/static/styles.css`
- `app/static/app.js`

Behavior:
- Uses `getUserMedia` + Web Audio API to capture mic audio.
- Encodes PCM audio into WAV in the browser.
- Sends audio via `multipart/form-data` to `/api/predict`.
- Renders the top emotion label, confidence, and full score list.
- Live streaming mode sends rolling 3s windows every ~1.5s while recording.

### Backend (FastAPI)

Files:
- `app/main.py`: FastAPI app, health check, static UI, and startup model loading.
- `app/model.py`: Loads `models/ser_lstm.keras` + `models/labels.json` and runs
  prediction.
- `app/audio.py`: Audio loading, MFCC extraction, and normalization.

Endpoints:
- `GET /` serves the web UI.
- `GET /api/health` health probe.
- `POST /api/predict` returns `{label, confidence, scores}`.

### Model training

Files:
- `train_model.py`: CLI training script matching notebook pipeline.
- `train_model_auto.py`: Auto-detects dataset path via `TESS_DATASET_DIR` or
  search under `--root`.

Outputs:
- `models/ser_lstm.keras`
- `models/labels.json`

### Deployment

Files:
- `render.yaml`: Render service definition.
- `Procfile`: Generic process start command.
