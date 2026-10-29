"""
check_setup.py - run this BEFORE the demo (works offline).

    python check_setup.py

Checks Python packages, model files, that every model actually loads,
that each feature produces output on the sample image, and whether a
webcam is available. Prints PASS / FAIL for each item.
"""

import importlib
import logging
import os
import sys
import time

OK, BAD, WARN = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m", "\033[93mWARN\033[0m"
if os.name == "nt":
    os.system("")  # enable ANSI colours on Windows terminals
fails = 0
logging.getLogger("streamlit").setLevel(logging.ERROR)


def check(label, fn, warn_only=False):
    global fails
    t0 = time.perf_counter()
    try:
        info = fn()
        print(f"[{OK}] {label}  {info or ''}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
    except Exception as exc:  # noqa: BLE001
        print(f"[{WARN if warn_only else BAD}] {label}: {exc}")
        if not warn_only:
            fails += 1


print(f"Python {sys.version.split()[0]}")
for pkg in ["cv2", "mediapipe", "numpy", "streamlit", "PIL", "pandas"]:
    check(f"import {pkg}", lambda p=pkg: getattr(importlib.import_module(p), "__version__", ""))
check("import streamlit_webrtc (optional, live tab)", lambda: importlib.import_module("streamlit_webrtc").__version__,
      warn_only=True)

from download_models import model_status, MODELS_DIR  # noqa: E402

for name, present in model_status().items():
    check(f"model file {name}", lambda p=present: None if p else (_ for _ in ()).throw(
        FileNotFoundError(f"missing in {MODELS_DIR} - run: python download_models.py")))

import cv2  # noqa: E402
import pipeline as P  # noqa: E402
import face_effects as fx  # noqa: E402

img = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images", "astronaut.png"))
M = P.Models()
check("BlazeFace detector", lambda: f"{len(M.detector.detect(img))} face(s) on sample")
check("Face mesh", lambda: f"{len(M.mesh.landmarks(img))} face(s), {M.mesh.landmarks(img)[0].shape[0]} landmarks")
check("Person segmenter", lambda: f"person area {M.segmenter.get_mask(img).mean():.0%}")
check("Haar cascade", lambda: f"{len(M.haar.detect(img))} face(s) on sample")
for preset in P.PRESETS:
    check(f"preset '{preset}'", lambda p=preset: f"{M.run(img, P.settings_from(P.PRESETS[p])).total_ms:.0f} ms")
check("head pose", lambda: "yaw {yaw:+.0f} pitch {pitch:+.0f} roll {roll:+.0f}".format(
    **fx.head_pose(M.mesh.landmarks(img)[0])))


def cam():
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no webcam frame (camera busy, missing or permission denied)")
    return f"{frame.shape[1]}x{frame.shape[0]}"


check("webcam 0", cam, warn_only=True)


def gui():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("no display found")
    cv2.namedWindow("t")
    cv2.destroyAllWindows()


check("OpenCV GUI (for live_camera.py)", gui, warn_only=True)

print("\nAll required checks passed - ready for the demo." if fails == 0 else f"\n{fails} check(s) failed - fix before the demo.")
sys.exit(1 if fails else 0)
