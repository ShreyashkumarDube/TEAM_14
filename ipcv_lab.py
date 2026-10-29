"""
ipcv_lab.py
----------------------------------------------------------------------
Classical image-processing operations from the IPCV syllabus, each
returning (output_image, short_explanation). Used by the "IPCV
Concepts Lab" tab to demonstrate the theory behind the app.
----------------------------------------------------------------------
"""

import cv2
import numpy as np


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


# ---------------- Colour spaces ----------------
def color_space(img, space: str):
    if space == "HSV":
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        chans, names = [h * (255 // 180), s, v], ["Hue", "Saturation", "Value"]
    elif space == "LAB":
        chans, names = list(cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))), ["L*", "a*", "b*"]
    elif space == "YCrCb":
        chans, names = list(cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))), ["Y (luma)", "Cr", "Cb"]
    else:
        b, g, r = cv2.split(img)
        chans, names = [r, g, b], ["Red", "Green", "Blue"]
    tiles = []
    for c, n in zip(chans, names):
        t = _bgr(c.astype(np.uint8))
        cv2.putText(t, n, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, n, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    return np.hstack(tiles), (f"Image split into {space} channels. HSV/LAB separate brightness from colour, "
                              "which is why the app edits brightness in V and enhances contrast in L*.")


# ---------------- Histogram processing ----------------
def histogram_equalization(img, method="Global", clip=2.0, tile=8):
    g = _gray(img)
    if method == "CLAHE":
        out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)
        txt = (f"CLAHE: equalisation in {tile}x{tile} tiles with histogram clip limit {clip}. "
               "Limits noise amplification and keeps local detail.")
    else:
        out = cv2.equalizeHist(g)
        txt = "Global equalisation: maps intensities through the CDF so the histogram becomes ~uniform."
    return _bgr(out), txt


# ---------------- Noise & denoising ----------------
def add_noise(img, kind="Gaussian", amount=20):
    rng = np.random.default_rng(0)
    if kind == "Salt & Pepper":
        out = img.copy()
        p = amount / 400.0
        r = rng.random(img.shape[:2])
        out[r < p / 2] = 0
        out[r > 1 - p / 2] = 255
        return out
    noise = rng.normal(0, amount, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def denoise(img, method="Median", k=5):
    k = k | 1
    if method == "Mean (box)":
        return cv2.blur(img, (k, k)), f"Box filter: each pixel = average of its {k}x{k} neighbourhood."
    if method == "Gaussian":
        return cv2.GaussianBlur(img, (k, k), 0), "Gaussian filter: weighted average, weights fall off with distance."
    if method == "Bilateral":
        return (cv2.bilateralFilter(img, k + 4, 75, 75),
                "Bilateral: weights depend on distance AND intensity difference, so edges are preserved.")
    if method == "Non-local Means":
        return (cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
                "Non-local means: averages similar patches from across the image.")
    return cv2.medianBlur(img, k), f"Median filter ({k}x{k}): non-linear, best for salt-and-pepper noise."


# ---------------- Edge detection ----------------
def edges(img, method="Sobel", low=80, high=160):
    g = cv2.GaussianBlur(_gray(img), (3, 3), 0)
    if method == "Sobel":
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        txt = "Sobel: first-derivative kernels Gx, Gy; edge strength = sqrt(Gx^2 + Gy^2)."
    elif method == "Prewitt":
        kx = np.array([[-1, 0, 1]] * 3, np.float32)
        gx = cv2.filter2D(g.astype(np.float32), -1, kx)
        gy = cv2.filter2D(g.astype(np.float32), -1, kx.T)
        mag = cv2.magnitude(gx, gy)
        txt = "Prewitt: like Sobel but with uniform (un-weighted) derivative kernels."
    elif method == "Laplacian":
        mag = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
        txt = "Laplacian: second derivative; responds to rapid intensity change in all directions."
    else:
        return _bgr(cv2.Canny(g, low, high)), (f"Canny: gradient + non-max suppression + hysteresis "
                                                f"thresholds ({low}, {high}) -> thin, connected edges.")
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return _bgr(mag), txt


# ---------------- Thresholding & morphology ----------------
def threshold(img, method="Otsu", value=127):
    g = _gray(img)
    if method == "Adaptive (Gaussian)":
        out = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        return _bgr(out), "Adaptive: a different threshold per 21x21 neighbourhood; handles uneven lighting."
    if method == "Manual":
        _, out = cv2.threshold(g, value, 255, cv2.THRESH_BINARY)
        return _bgr(out), f"Global threshold at T = {value}."
    t, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _bgr(out), f"Otsu: automatically picks T = {t:.0f}, maximising between-class variance."


def morphology(img, op="Opening", k=5, iterations=1):
    _, binary = cv2.threshold(_gray(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ops = {"Erosion": cv2.MORPH_ERODE, "Dilation": cv2.MORPH_DILATE, "Opening": cv2.MORPH_OPEN,
           "Closing": cv2.MORPH_CLOSE, "Gradient": cv2.MORPH_GRADIENT, "Top-hat": cv2.MORPH_TOPHAT}
    out = cv2.morphologyEx(binary, ops[op], se, iterations=iterations)
    desc = {"Erosion": "shrinks white regions", "Dilation": "grows white regions",
            "Opening": "erosion then dilation - removes small white specks",
            "Closing": "dilation then erosion - fills small black holes",
            "Gradient": "dilation minus erosion - object outlines",
            "Top-hat": "image minus opening - small bright details"}[op]
    return np.hstack([_bgr(binary), _bgr(out)]), f"{op} with {k}x{k} elliptical element on the Otsu mask: {desc}."


# ---------------- Frequency domain ----------------
def fft_filter(img, kind="Low-pass", radius=30, butterworth_order=2):
    g = _gray(img).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g))
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    D = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    H = 1.0 / (1.0 + (D / max(radius, 1)) ** (2 * butterworth_order))   # Butterworth low-pass
    if kind == "High-pass":
        H = 1.0 - H
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    if kind == "High-pass":
        out = np.abs(out)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum = np.log1p(np.abs(F))
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filt = (H * 255).astype(np.uint8)
    tiles = [_bgr(spectrum), cv2.applyColorMap(filt, cv2.COLORMAP_VIRIDIS), _bgr(out)]
    for t, n in zip(tiles, ["log |F(u,v)|", f"{kind} H(u,v)", "Filtered"]):
        cv2.putText(t, n, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, n, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack(tiles), (f"Butterworth {kind.lower()} filter (D0 = {radius}, order {butterworth_order}) "
                              "applied in the frequency domain: F -> F*H -> inverse FFT. "
                              "Low-pass = blur, high-pass = edges/detail.")


# ---------------- Geometric transforms ----------------
def geometric(img, angle=15, scale=1.0, flip="None"):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if flip == "Horizontal":
        out = cv2.flip(out, 1)
    elif flip == "Vertical":
        out = cv2.flip(out, 0)
    return out, (f"Affine transform: rotation {angle} deg, scale {scale}, bilinear interpolation. "
                 f"Matrix M = [[{M[0,0]:.2f}, {M[0,1]:.2f}, {M[0,2]:.0f}], [{M[1,0]:.2f}, {M[1,1]:.2f}, {M[1,2]:.0f}]].")
