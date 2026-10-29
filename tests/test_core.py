"""Unit tests - run with:  python -m pytest -q"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import filters  # noqa: E402
import ipcv_lab as lab  # noqa: E402
import metrics  # noqa: E402
import pipeline as P  # noqa: E402
from download_models import model_status  # noqa: E402

RNG = np.random.default_rng(0)
IMG = RNG.integers(0, 256, (120, 160, 3), dtype=np.uint8)


def test_every_style_keeps_shape_and_dtype():
    for name, fn in filters.STYLES.items():
        if fn is None:
            continue
        out = fn(IMG)
        assert out.shape == IMG.shape and out.dtype == np.uint8, name


def test_zero_adjustments_are_identity():
    s = P.settings_from()
    r = P.run_pipeline(IMG, s)
    assert np.array_equal(r.image, IMG) and r.timings == []


def test_brightness_direction():
    assert filters.adjust_brightness(IMG, 40).mean() > IMG.mean()
    assert filters.adjust_brightness(IMG, -40).mean() < IMG.mean()


def test_metrics_identity_and_ranges():
    assert metrics.psnr(IMG, IMG) == float("inf")
    assert metrics.ssim(IMG, IMG) == pytest.approx(1.0, abs=1e-6)
    noisy = lab.add_noise(IMG, "Gaussian", 30)
    assert 0 < metrics.ssim(IMG, noisy) < 1
    assert metrics.psnr(IMG, noisy) < 30


def test_background_composite_uses_mask():
    mask = np.zeros(IMG.shape[:2], np.float32)
    mask[:, :80] = 1.0  # left half = person
    out = filters.replace_background_color(IMG, mask, (0, 0, 255), feather=0)
    assert np.array_equal(out[:, :80], IMG[:, :80])
    assert (out[:, 81:] == (0, 0, 255)).all()


def test_ipcv_lab_functions_run():
    lab.color_space(IMG, "HSV")
    lab.histogram_equalization(IMG, "CLAHE")
    lab.denoise(lab.add_noise(IMG, "Salt & Pepper", 30), "Median", 5)
    for m in ["Sobel", "Prewitt", "Laplacian", "Canny"]:
        lab.edges(IMG, m)
    lab.threshold(IMG, "Otsu")
    lab.morphology(IMG, "Opening")
    lab.fft_filter(IMG, "High-pass")
    lab.geometric(IMG, 30, 1.0, "Horizontal")


def test_presets_only_use_known_keys():
    for name, preset in P.PRESETS.items():
        assert set(preset) <= set(P.DEFAULTS), name


@pytest.mark.skipif(not all(model_status().values()), reason="model files not downloaded")
def test_models_on_sample_image():
    import cv2
    import face_effects as fx
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img = cv2.imread(os.path.join(here, "sample_images", "astronaut.png"))
    M = P.Models()
    assert len(M.detector.detect(img)) == 1
    lms = M.mesh.landmarks(img)
    assert len(lms) == 1 and lms[0].shape == (478, 3)
    pose = fx.head_pose(lms[0])
    assert abs(pose["yaw"]) < 25 and abs(pose["roll"]) < 25
    assert 0.15 < fx.eye_state(lms[0])["ear"] < 0.5
    mask = M.segmenter.get_mask(img)
    assert mask.shape == img.shape[:2] and 0.2 < mask.mean() < 0.9
    for preset in P.PRESETS.values():
        assert M.run(img, P.settings_from(preset)).image.shape == img.shape
