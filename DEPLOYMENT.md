# Render Deployment Guide

## Prerequisites

- A trained model saved at `models/ser_lstm.keras`.
- Labels saved at `models/labels.json`.
- Repo pushed to GitHub.

## Option A: Blueprint Deploy (recommended)

1) Ensure `render.yaml` is in the repo root.
2) Push the repo to GitHub.
3) In Render, choose **New +** → **Blueprint**.
4) Connect the repo and let Render read `render.yaml`.
5) Deploy.

Render will run:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Option B: Manual Web Service

1) Create a **Web Service** from this repo.
2) Set Build Command:
   - `pip install -r requirements.txt`
3) Set Start Command:
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4) Deploy.

## Model files on Render

The API will fail if the model files are missing. Make sure:

- `models/ser_lstm.keras`
- `models/labels.json`

are present in the repository (or added during build).

## Notes

- Render free instances sleep; first request can be slow.
- If you retrain the model locally, commit the updated model files before
  redeploying.
