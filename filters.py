"""
filters.py
----------------------------------------------------------------------
Point operations, neighbourhood filters, segmentation-based compositing
and artistic styles built with OpenCV + NumPy.

Every function takes and returns a BGR uint8 image (OpenCV convention)
so filters can be chained freely inside the processing pipeline.
----------------------------------------------------------------------
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Tonal adjustments (point operations)
# ----------------------------------------------------------------------
def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
    """value in [-100, 100]: additive offset on the HSV Value channel."""
    if value == 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(image: np.ndarray, value: float) -> np.ndarray:
    """value in [-100, 100]: linear stretch about mid-grey, factor = 259(v+255)/(255(259-v))."""
    if value == 0:
        return image
    factor = (259 * (value + 255)) / (255 * (259 - value))
    out = (image.astype(np.float32) - 128) * factor + 128
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_saturation(image: np.ndarray, value: int) -> np.ndarray:
    """value in [-100, 100]: multiplicative scale of the HSV Saturation channel."""
    if value == 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + value / 100.0), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law transform s = 255 * (r/255)^(1/gamma) via a 256-entry lookup table."""
    if abs(gamma - 1.0) < 1e-3:
        return image
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def adjust_sharpness(image: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp masking: out = (1+a)*I - a*Gaussian(I). a<0 softens."""
    if amount == 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def adjust_warmth(image: np.ndarray, value: int) -> np.ndarray:
    """value in [-100, 100]: +R/-B for warm, -R/+B for cool."""
    if value == 0:
        return image
    img = image.astype(np.int16)
    img[:, :, 2] = np.clip(img[:, :, 2] + value, 0, 255)
    img[:, :, 0] = np.clip(img[:, :, 0] - value, 0, 255)
    return img.astype(np.uint8)


def auto_enhance(image: np.ndarray) -> np.ndarray:
    """One-click enhance: CLAHE on the L channel of LAB + mild saturation boost."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return adjust_saturation(out, 12)


# ----------------------------------------------------------------------
# Segmentation-based compositing (portrait mode)
# ----------------------------------------------------------------------
def _prepare_mask(mask: np.ndarray, shape, feather: int) -> np.ndarray:
    h, w = shape[:2]
    m = cv2.resize(mask.astype(np.float32), (w, h)) if mask.shape[:2] != (h, w) else mask.astype(np.float32)
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return np.clip(m, 0, 1)[..., None]


def alpha_composite(fg: np.ndarray, bg: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """out = alpha*fg + (1-alpha)*bg (alpha = soft person mask)."""
    out = fg.astype(np.float32) * mask3 + bg.astype(np.float32) * (1 - mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def blur_background(image: np.ndarray, mask: np.ndarray, strength: int = 15, feather: int = 15) -> np.ndarray:
    """Portrait-mode bokeh: Gaussian-blurred background, sharp person."""
    k = max(strength, 1) * 2 + 1
    return alpha_composite(image, cv2.GaussianBlur(image, (k, k), 0), _prepare_mask(mask, image.shape, feather))


def replace_background_color(image: np.ndarray, mask: np.ndarray, color=(40, 40, 40), feather: int = 15) -> np.ndarray:
    solid = np.full_like(image, color, dtype=np.uint8)
    return alpha_composite(image, solid, _prepare_mask(mask, image.shape, feather))


def replace_background_image(image: np.ndarray, mask: np.ndarray, background: np.ndarray, feather: int = 15) -> np.ndarray:
    """Replace background with another photo (resized with centre-crop to fill)."""
    h, w = image.shape[:2]
    bh, bw = background.shape[:2]
    scale = max(w / bw, h / bh)
    bg = cv2.resize(background, (int(np.ceil(bw * scale)), int(np.ceil(bh * scale))))
    y0, x0 = (bg.shape[0] - h) // 2, (bg.shape[1] - w) // 2
    bg = bg[y0:y0 + h, x0:x0 + w]
    return alpha_composite(image, bg, _prepare_mask(mask, image.shape, feather))


def greyscale_background(image: np.ndarray, mask: np.ndarray, feather: int = 15) -> np.ndarray:
    """'Colour splash': person stays in colour, background goes grey."""
    return alpha_composite(image, grayscale(image), _prepare_mask(mask, image.shape, feather))


# ----------------------------------------------------------------------
# Artistic styles
# ----------------------------------------------------------------------
def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def sepia(image: np.ndarray) -> np.ndarray:
    # Standard sepia matrix. Rows give B', G', R' (output in BGR order);
    # columns are written for (R, G, B) input, so reverse them for BGR input.
    kernel = np.array([[0.272, 0.534, 0.131],   # B'
                       [0.349, 0.686, 0.168],   # G'
                       [0.393, 0.769, 0.189]])  # R'
    return cv2.transform(image, kernel[:, ::-1])


def cartoonize(image: np.ndarray) -> np.ndarray:
    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = image
    for _ in range(2):  # repeated bilateral = flatter "cel" colours
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
    # colour quantisation to 8 levels per channel
    color = (color // 32) * 32 + 16
    return cv2.bitwise_and(color, color, mask=edges)


def pencil_sketch(image: np.ndarray) -> np.ndarray:
    """Colour-dodge blend of greyscale with its blurred negative."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv_blur = 255 - cv2.GaussianBlur(255 - gray, (21, 21), 0)
    return cv2.cvtColor(cv2.divide(gray, inv_blur, scale=256.0), cv2.COLOR_GRAY2BGR)


def color_sketch(image: np.ndarray) -> np.ndarray:
    _, col = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return col


def watercolor(image: np.ndarray) -> np.ndarray:
    return cv2.stylization(image, sigma_s=60, sigma_r=0.45)


def oil_paint(image: np.ndarray) -> np.ndarray:
    """Approximation via edge-preserving filter + detail enhancement."""
    smooth = cv2.edgePreservingFilter(image, flags=cv2.RECURS_FILTER, sigma_s=50, sigma_r=0.4)
    return cv2.detailEnhance(smooth, sigma_s=10, sigma_r=0.15)


def edge_detect(image: np.ndarray, low: int = 80, high: int = 160) -> np.ndarray:
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0), low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def emboss(image: np.ndarray) -> np.ndarray:
    k = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    return cv2.filter2D(image, -1, k, delta=0)


def noir(image: np.ndarray) -> np.ndarray:
    """High-contrast black & white with crushed blacks."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    g = adjust_contrast(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 40)
    return g


def thermal(image: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)


def cool_tone(image: np.ndarray) -> np.ndarray:
    return adjust_warmth(image, -40)


def warm_tone(image: np.ndarray) -> np.ndarray:
    return adjust_warmth(image, 40)


STYLES = {
    "None": None,
    "Grayscale": grayscale,
    "Sepia": sepia,
    "Noir (high-contrast B&W)": noir,
    "Cartoon": cartoonize,
    "Pencil Sketch": pencil_sketch,
    "Colour Pencil": color_sketch,
    "Watercolour": watercolor,
    "Oil Paint": oil_paint,
    "Emboss": emboss,
    "Edge Detect (Canny)": edge_detect,
    "Thermal (colour map)": thermal,
    "Cool Tone": cool_tone,
    "Warm Tone": warm_tone,
}


# ----------------------------------------------------------------------
# Finishing effects
# ----------------------------------------------------------------------
def apply_vignette(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Multiply by a separable 2-D Gaussian; higher strength = smaller sigma = darker corners."""
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    kx = cv2.getGaussianKernel(w, w / (0.5 * strength))
    ky = cv2.getGaussianKernel(h, h / (0.5 * strength))
    mask = ky * kx.T
    mask = (mask / mask.max())[..., None]
    return np.clip(image.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def add_film_grain(image: np.ndarray, amount: float = 12) -> np.ndarray:
    if amount <= 0:
        return image
    noise = np.random.default_rng(7).normal(0, amount, image.shape[:2]).astype(np.float32)[..., None]
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def resize_max_dim(image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    s = max_dim / max(h, w)
    return cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def hex_to_bgr(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)
