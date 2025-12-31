import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model

from train_model import collect_dataset
from app.audio import extract_mfcc_from_path

matplotlib.use("Agg")


def load_labels(labels_path: Path) -> list[str]:
    return json.loads(labels_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SER model and save plots.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the TESS dataset root directory.",
    )
    parser.add_argument("--model-path", default="models/ser_lstm.keras")
    parser.add_argument("--labels-path", default="models/labels.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    labels_path = Path(args.labels_path)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    paths, labels = collect_dataset(data_dir)
    if not paths:
        raise RuntimeError("No WAV files found in the dataset directory.")

    features = [extract_mfcc_from_path(path) for path in paths]
    x = np.expand_dims(np.array(features), -1)
    y = np.array(labels)

    x_train, x_test, y_train, y_test, paths_train, paths_test = train_test_split(
        x,
        y,
        np.array(paths),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    model = load_model(model_path)
    label_list = load_labels(labels_path)
    label_to_idx = {label: i for i, label in enumerate(label_list)}

    y_test_idx = np.array([label_to_idx[label] for label in y_test])
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_test_idx, y_pred_idx, target_names=label_list, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(results_dir / "metrics.csv", index=True)
    report_df.to_markdown(results_dir / "metrics.md", index=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test_idx,
        y_pred_idx,
        display_labels=label_list,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    y_test_bin = label_binarize(y_test_idx, classes=list(range(len(label_list))))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(label_list):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "roc_curves.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(label_list):
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, i], y_pred_probs[:, i]
        )
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "pr_curves.png", dpi=300)
    plt.close(fig)

    predictions = pd.DataFrame(
        {
            "path": paths_test,
            "true_label": y_test,
            "pred_label": [label_list[i] for i in y_pred_idx],
        }
    )
    predictions["correct"] = predictions["true_label"] == predictions["pred_label"]
    predictions.head(50).to_csv(results_dir / "sample_predictions.csv", index=False)

    history_path = Path("models/history.json")
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = list(range(1, len(history.get("accuracy", [])) + 1))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, history.get("accuracy", []), label="train")
        ax.plot(epochs, history.get("val_accuracy", []), label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "training_accuracy.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, history.get("loss", []), label="train")
        ax.plot(epochs, history.get("val_loss", []), label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "training_loss.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
