"""
metrics.py
----------------------------------------------------------------------
Objective image-quality and statistics measures used in the Results
panel and in benchmark.py.

    psnr(a, b)          Peak signal-to-noise ratio (dB)
    ssim(a, b)          Structural similarity (Wang et al., 2004), 0..1
    image_stats(img)    brightness, contrast, entropy, sharpness, colourfulness
    histograms(img)     256-bin histograms per channel
----------------------------------------------------------------------
"""

import cv2
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on the luminance channel with an 11x11 Gaussian window (sigma 1.5)."""
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    s_aa = blur(a * a) - mu_a ** 2
    s_bb = blur(b * b) - mu_b ** 2
    s_ab = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2)
    return float(np.mean(num / den))


def entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def colourfulness(img: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric."""
    b, g, r = [c.astype(np.float64) for c in cv2.split(img)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def image_stats(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "Brightness (mean)": float(gray.mean()),
        "Contrast (std)": float(gray.std()),
        "Entropy (bits)": entropy(gray),
        "Sharpness (Laplacian var)": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "Colourfulness": colourfulness(img),
    }


def compare(original: np.ndarray, processed: np.ndarray) -> dict:
    return {"PSNR (dB)": psnr(original, processed), "SSIM": ssim(original, processed)}


def histograms(img: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(["Blue", "Green", "Red"]):
        out[name] = cv2.calcHist([img], [i], None, [256], [0, 256]).ravel()
    return out
