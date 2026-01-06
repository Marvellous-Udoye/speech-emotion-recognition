import json
import time
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
from tensorflow.keras.models import load_model

from .audio import extract_mfcc_from_audio, load_audio_bytes

MODEL_PATH = Path("models/ser_lstm.keras")
LABELS_PATH = Path("models/labels.json")


class EmotionModel:
    def __init__(self, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train it with train_model.py first."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels not found at {labels_path}. Train it with train_model.py first."
            )

        self.model = load_model(model_path)
        self.labels = json.loads(labels_path.read_text(encoding="utf-8"))

    def predict(self, audio_bytes: bytes) -> Dict[str, float]:
        start = time.time()
        try:
            y, sr = load_audio_bytes(audio_bytes)
            mfcc = extract_mfcc_from_audio(y, sr)
            x = np.expand_dims(mfcc, axis=(0, -1))
            print(f"[model] input_shape={x.shape}", flush=True)
            probs = self.model.predict(x, verbose=0)[0]
            scores = {label: float(prob) for label, prob in zip(self.labels, probs)}
            best_idx = int(np.argmax(probs))
            return {
                "label": self.labels[best_idx],
                "confidence": float(probs[best_idx]),
                "scores": scores,
            }
        except Exception as exc:
            print(f"[model] error: {exc}", flush=True)
            traceback.print_exc()
            raise
        finally:
            elapsed_ms = int((time.time() - start) * 1000)
            print(f"[model] predict_elapsed_ms={elapsed_ms}", flush=True)
