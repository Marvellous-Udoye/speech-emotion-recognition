import io
from typing import Tuple

import librosa
import numpy as np

TARGET_SR = 22050
DURATION_SEC = 3.0
OFFSET_SEC = 0.5
N_MFCC = 40


def _fix_length(y: np.ndarray, sr: int) -> np.ndarray:
    start = int(OFFSET_SEC * sr)
    if y.shape[0] > start:
        y = y[start:]
    target_len = int(DURATION_SEC * sr)
    return librosa.util.fix_length(y, size=target_len)


def load_audio_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    print(f"[audio] bytes={len(data)} target_sr={TARGET_SR}", flush=True)
    with io.BytesIO(data) as bio:
        y, sr = librosa.load(bio, sr=TARGET_SR, mono=True)
    print(f"[audio] raw_shape={y.shape} sr={sr}", flush=True)
    y = _fix_length(y, sr)
    print(f"[audio] fixed_shape={y.shape}", flush=True)
    return y, sr


def extract_mfcc_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    print(f"[audio] mfcc_shape={mfcc.shape}", flush=True)
    return np.mean(mfcc.T, axis=0)


def extract_mfcc_from_path(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=TARGET_SR, duration=DURATION_SEC, offset=OFFSET_SEC)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)
