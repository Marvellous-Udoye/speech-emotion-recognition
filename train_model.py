import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

from app.audio import extract_mfcc_from_path


def collect_dataset(data_dir: Path) -> tuple[list[str], list[str]]:
    paths = []
    labels = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if not filename.lower().endswith(".wav"):
                continue
            paths.append(os.path.join(dirname, filename))
            label = filename.split("_")[-1].split(".")[0].lower()
            labels.append(label)
    return paths, labels


def build_model() -> Sequential:
    model = Sequential(
        [
            Input(shape=(40, 1)),
            LSTM(256, return_sequences=False),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(7, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def run_training(
    data_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    model_out: Path = Path("models/ser_lstm.keras"),
    labels_out: Path = Path("models/labels.json"),
) -> None:
    paths, labels = collect_dataset(data_dir)
    if not paths:
        raise RuntimeError("No WAV files found in the dataset directory.")

    features = [extract_mfcc_from_path(path) for path in paths]
    x = np.expand_dims(np.array(features), -1)

    enc = OneHotEncoder()
    y = enc.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    label_list = [label for label in enc.categories_[0]]

    model = build_model()
    history = model.fit(
        x,
        y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_out)
    labels_out.write_text(json.dumps(label_list, indent=2), encoding="utf-8")

    best_val = max(history.history.get("val_accuracy", [0]))
    print(f"Training complete. Best val accuracy: {best_val:.4f}")
    print(f"Saved model to {model_out}")
    print(f"Saved labels to {labels_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SER LSTM model.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the TESS dataset root directory.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--model-out",
        default="models/ser_lstm.keras",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--labels-out",
        default="models/labels.json",
        help="Where to save label names.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    run_training(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_out=Path(args.model_out),
        labels_out=Path(args.labels_out),
    )


if __name__ == "__main__":
    main()
