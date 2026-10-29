"""
download_models.py
----------------------------------------------------------------------
Downloads the MediaPipe model files used by Face Studio Camera into
./models/ so the app can run fully OFFLINE afterwards (important for
lab demos where Wi-Fi may be unavailable).

    python download_models.py            # download anything missing
    python download_models.py --force    # re-download everything

The Haar cascade XML (used only for the Haar-vs-MediaPipe comparison)
is shipped inside ./models/ already, so it needs no download.

Set the environment variable FSC_MODELS_DIR to use a different folder.
----------------------------------------------------------------------
"""

import os
import sys
import urllib.request

MODELS_DIR = os.environ.get(
    "FSC_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
)

MODELS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "selfie_segmenter.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
}

HAAR_FILE = "haarcascade_frontalface_default.xml"


class ModelMissingError(RuntimeError):
    """Raised when a model is not on disk and cannot be downloaded."""


def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        sys.stdout.write(f"\r    {pct:3d}%")
        sys.stdout.flush()


def model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def model_status() -> dict:
    """{filename: True/False} for every model the app uses."""
    names = list(MODELS) + [HAAR_FILE]
    return {n: os.path.exists(model_path(n)) for n in names}


def _download(filename: str, dest: str) -> None:
    tmp = dest + ".part"
    urllib.request.urlretrieve(MODELS[filename], tmp, reporthook=_reporthook)
    os.replace(tmp, dest)  # atomic: never leaves a half-written model behind


def download_all(force: bool = False) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename in MODELS:
        dest = model_path(filename)
        if os.path.exists(dest) and not force:
            print(f"[skip] {filename} already exists")
            continue
        print(f"[download] {filename}")
        try:
            _download(filename, dest)
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"\n[error] Failed to download {filename}: {exc}")
            print(f"        Download it manually from:\n        {MODELS[filename]}")
            print(f"        and place it at:\n        {dest}")
            sys.exit(1)
    if not os.path.exists(model_path(HAAR_FILE)):
        print(f"[warn] {HAAR_FILE} is missing from models/ (needed for Haar comparison).")
    print(f"\nAll models are ready in {MODELS_DIR}")


def ensure_model(filename: str) -> str:
    """Return the local path of a model, downloading it once if missing."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = model_path(filename)
    if os.path.exists(dest):
        return dest
    if filename not in MODELS:
        raise ModelMissingError(f"Model file '{filename}' not found in {MODELS_DIR}.")
    try:
        print(f"Downloading required model '{filename}' (first run only)...")
        _download(filename, dest)
        print()
        return dest
    except Exception as exc:  # noqa: BLE001
        raise ModelMissingError(
            f"Model '{filename}' is not in {MODELS_DIR} and could not be downloaded "
            f"({exc}). Connect to the internet and run: python download_models.py"
        ) from exc


if __name__ == "__main__":
    download_all(force="--force" in sys.argv)
