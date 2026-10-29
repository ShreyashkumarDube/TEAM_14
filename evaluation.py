"""
evaluation.py
----------------------------------------------------------------------
Haar cascade vs MediaPipe BlazeFace robustness experiment.

Each test image is transformed by a set of controlled degradations
(rotation, low light, over-exposure, noise, blur, down-scaling,
horizontal flip). Both detectors run on every variant; we record
whether at least one face was found, how many boxes were returned
(extra boxes = likely false positives) and the run-time.

Used by the Face Analysis tab and by benchmark.py.
----------------------------------------------------------------------
"""

import time

import cv2
import numpy as np


def _rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _gamma(img, g):
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def _noise(img, sigma):
    n = np.random.default_rng(1).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _small(img, s):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


VARIANTS = {
    "Original": lambda x: x,
    "Rotated 15 deg": lambda x: _rotate(x, 15),
    "Rotated 30 deg": lambda x: _rotate(x, 30),
    "Rotated 45 deg": lambda x: _rotate(x, 45),
    "Low light (gamma 3)": lambda x: _gamma(x, 3.0),
    "Very dark (gamma 5)": lambda x: _gamma(x, 5.0),
    "Over-exposed (gamma 0.3)": lambda x: _gamma(x, 0.3),
    "Gaussian noise (sigma 25)": lambda x: _noise(x, 25),
    "Gaussian noise (sigma 50)": lambda x: _noise(x, 50),
    "Motion/defocus blur (15 px)": lambda x: cv2.GaussianBlur(x, (15, 15), 0),
    "Down-scaled to 40%": lambda x: _small(x, 0.4),
    "Mirrored": lambda x: cv2.flip(x, 1),
}


class SinglePass:
    """Wraps FaceDetector so detect() uses one full-frame pass (no tiling)."""

    def __init__(self, detector):
        self.d = detector

    def detect(self, img):
        return self.d.detect(img, multiscale=False)


def run_detector(detector, img, repeats=3):
    detector.detect(img)                      # warm-up (first call allocates buffers)
    t0 = time.perf_counter()
    for _ in range(repeats):
        faces = detector.detect(img)
    return faces, (time.perf_counter() - t0) * 1000 / repeats


def robustness_table(img, mp_detector, haar_detector, expected_faces=1, variants=None, repeats=3):
    """Return list of dict rows, one per variant."""
    rows = []
    for name, fn in (variants or VARIANTS).items():
        v = fn(img)
        mp_f, mp_ms = run_detector(mp_detector, v, repeats)
        sp_f, sp_ms = run_detector(SinglePass(mp_detector), v, repeats)
        hr_f, hr_ms = run_detector(haar_detector, v, repeats)
        rows.append({
            "Condition": name,
            "Expected": expected_faces,
            "MediaPipe faces": len(mp_f),
            "MediaPipe OK": len(mp_f) >= expected_faces,
            "MediaPipe ms": round(mp_ms, 1),
            "MediaPipe conf": round(max((f["score"] for f in mp_f), default=0.0), 2),
            "Single-pass faces": len(sp_f),
            "Single-pass OK": len(sp_f) >= expected_faces,
            "Single-pass ms": round(sp_ms, 1),
            "Haar faces": len(hr_f),
            "Haar OK": len(hr_f) >= expected_faces,
            "Haar ms": round(hr_ms, 1),
        })
    return rows


def summarise(rows, expected_faces=1):
    n = len(rows)

    def fp(key):
        return sum(max(0, r[key] - r.get("Expected", expected_faces)) for r in rows)
    return {
        "Conditions tested": n,
        "MediaPipe detection rate": f"{100 * sum(r['MediaPipe OK'] for r in rows) / n:.0f}%",
        "Single-pass detection rate": f"{100 * sum(r['Single-pass OK'] for r in rows) / n:.0f}%",
        "Haar detection rate": f"{100 * sum(r['Haar OK'] for r in rows) / n:.0f}%",
        "MediaPipe extra boxes": fp("MediaPipe faces"),
        "Single-pass extra boxes": fp("Single-pass faces"),
        "Haar extra boxes": fp("Haar faces"),
        "MediaPipe mean ms": round(float(np.mean([r["MediaPipe ms"] for r in rows])), 1),
        "Single-pass mean ms": round(float(np.mean([r["Single-pass ms"] for r in rows])), 1),
        "Haar mean ms": round(float(np.mean([r["Haar ms"] for r in rows])), 1),
    }
