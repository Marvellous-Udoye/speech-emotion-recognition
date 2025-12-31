import argparse
import os
from pathlib import Path
from typing import Optional

from train_model import run_training


def _has_wav_files(path: Path) -> bool:
    try:
        return any(p.suffix.lower() == ".wav" for p in path.rglob("*.wav"))
    except PermissionError:
        return False


def find_dataset(root: Path) -> Optional[Path]:
    candidates = []
    known_names = [
        "TESS Toronto emotional speech set data",
        "TESS",
        "tess",
        "toronto emotional speech set",
    ]

    for name in known_names:
        cand = root / name
        if cand.exists():
            candidates.append(cand)

    for path in root.iterdir():
        if path.is_dir() and "tess" in path.name.lower():
            candidates.append(path)

    for cand in candidates:
        if _has_wav_files(cand):
            return cand

    for cand in root.rglob("*"):
        if cand.is_dir() and "tess" in cand.name.lower():
            if _has_wav_files(cand):
                return cand
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-detect TESS dataset and train.")
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to search for the dataset.",
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

    root = Path(args.root).resolve()
    env_path = os.environ.get("TESS_DATASET_DIR")
    if env_path:
        data_dir = Path(env_path).resolve()
    else:
        data_dir = find_dataset(root)

    if data_dir is None or not data_dir.exists():
        raise FileNotFoundError(
            "Could not auto-detect the dataset. Set TESS_DATASET_DIR or use "
            "--root to point to the parent folder."
        )

    print(f"Using dataset: {data_dir}")
    run_training(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_out=Path(args.model_out),
        labels_out=Path(args.labels_out),
    )


if __name__ == "__main__":
    main()
