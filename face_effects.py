"""
face_effects.py
----------------------------------------------------------------------
Landmark-driven face effects and measurements.

    head_pose()            yaw / pitch / roll from 3-D landmarks
    eye_aspect_ratio()     blink / eye-openness (EAR)
    mouth_aspect_ratio()   mouth-open detection (MAR)
    draw_ar_overlay()      sunglasses, party hat, moustache, cat face
    anonymize_faces()      privacy blur / pixelate / eye-bar
    beautify()             skin smoothing restricted to skin pixels
    tint_lips(), tint_eyes()

All AR overlays are drawn procedurally (no image assets) in a
FACE-LOCAL coordinate frame:
    origin = midpoint of the two eye centres
    unit   = inter-ocular distance d
    u-axis = from the subject's right eye to the left eye
    v-axis = perpendicular, pointing towards the chin
so every overlay automatically follows the face's position, size and
in-plane rotation (roll).
----------------------------------------------------------------------
"""

import cv2
import numpy as np

from face_utils import (EAR_LEFT, EAR_RIGHT, FACE_OVAL, LEFT_BROW, LEFT_EYE, LEFT_IRIS,
                        LIPS_INNER, LIPS_OUTER, RIGHT_BROW, RIGHT_EYE, RIGHT_IRIS)


# ----------------------------------------------------------------------
# Face-local coordinate frame
# ----------------------------------------------------------------------
class FaceFrame:
    def __init__(self, lm: np.ndarray):
        self.lm = lm[:, :2]
        self.right_eye = self.lm[RIGHT_EYE].mean(axis=0)   # appears on image LEFT
        self.left_eye = self.lm[LEFT_EYE].mean(axis=0)
        self.origin = (self.right_eye + self.left_eye) / 2
        v = self.left_eye - self.right_eye
        self.d = float(np.linalg.norm(v)) + 1e-6
        self.ex = v / self.d
        self.ey = np.array([-self.ex[1], self.ex[0]])
        self.roll_deg = float(np.degrees(np.arctan2(self.ex[1], self.ex[0])))

    def to_img(self, uv) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float32))
        pts = self.origin + self.d * (uv[:, :1] * self.ex + uv[:, 1:2] * self.ey)
        return pts.astype(np.int32)

    def to_local(self, idx: int) -> np.ndarray:
        p = self.lm[idx] - self.origin
        return np.array([p @ self.ex, p @ self.ey]) / self.d


def _ellipse_uv(cu, cv, ru, rv, n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cu + ru * np.cos(t), cv + rv * np.sin(t)], axis=1)


def _blend(base, layer, alpha_mask):
    a = alpha_mask.astype(np.float32)[..., None]
    return np.clip(base.astype(np.float32) * (1 - a) + layer.astype(np.float32) * a, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# AR overlays
# ----------------------------------------------------------------------
def _sunglasses(img, f: FaceFrame):
    layer = img.copy()
    alpha = np.zeros(img.shape[:2], np.float32)
    lenses = []
    for cu in (-0.5, 0.5):
        uv = _ellipse_uv(cu, 0.03, 0.47, 0.33)
        uv[:, 1] = np.where(uv[:, 1] < 0.02, 0.02 + (uv[:, 1] - 0.02) * 0.75, uv[:, 1])  # flatter top
        lenses.append(f.to_img(uv))
    for poly in lenses:
        cv2.fillPoly(layer, [poly], (25, 20, 15), cv2.LINE_AA)
        cv2.fillPoly(alpha, [poly], 0.88, cv2.LINE_AA)
    th = max(2, int(f.d * 0.06))
    for poly in lenses:
        cv2.polylines(layer, [poly], True, (10, 10, 10), th, cv2.LINE_AA)
        cv2.polylines(alpha, [poly], True, 1.0, th, cv2.LINE_AA)
    bridge = f.to_img([[-0.08, -0.12], [0, -0.16], [0.08, -0.12]])
    arms = [f.to_img([[-0.93, -0.08], [-1.08, -0.05]]), f.to_img([[0.93, -0.08], [1.08, -0.05]])]
    for line in [bridge] + arms:
        cv2.polylines(layer, [line], False, (10, 10, 10), th, cv2.LINE_AA)
        cv2.polylines(alpha, [line], False, 1.0, th, cv2.LINE_AA)
    out = _blend(img, layer, alpha)
    # specular glare
    glare = out.copy()
    ga = np.zeros(img.shape[:2], np.float32)
    for cu in (-0.5, 0.5):
        g = f.to_img([[cu - 0.25, -0.05], [cu - 0.1, -0.15]])
        cv2.line(glare, tuple(g[0]), tuple(g[1]), (255, 255, 255), max(2, th // 2), cv2.LINE_AA)
        cv2.line(ga, tuple(g[0]), tuple(g[1]), 0.6, max(2, th // 2), cv2.LINE_AA)
    return _blend(out, glare, ga)


def _party_hat(img, f: FaceFrame):
    top_v = f.to_local(10)[1]            # landmark 10 = top of forehead
    base_v, apex_v = top_v + 0.05, top_v - 1.7
    layer = img.copy()
    alpha = np.zeros(img.shape[:2], np.float32)
    tri = f.to_img([[-0.75, base_v], [0.75, base_v], [0.0, apex_v]])
    cv2.fillPoly(layer, [tri], (200, 80, 230), cv2.LINE_AA)
    cv2.fillPoly(alpha, [tri], 1.0, cv2.LINE_AA)
    # diagonal stripes (clipped to the triangle)
    stripes = np.zeros(img.shape[:2], np.uint8)
    for k in range(-6, 8):
        s = f.to_img([[-1.2 + 0.3 * k, base_v], [-0.2 + 0.3 * k, apex_v]])
        cv2.line(stripes, tuple(s[0]), tuple(s[1]), 255, max(3, int(f.d * 0.09)), cv2.LINE_AA)
    tri_mask = np.zeros_like(stripes)
    cv2.fillPoly(tri_mask, [tri], 255)
    layer[(stripes > 0) & (tri_mask > 0)] = (60, 220, 255)
    pom = f.to_img([[0, apex_v]])[0]
    r = max(4, int(f.d * 0.16))
    cv2.circle(layer, tuple(pom), r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(alpha, tuple(pom), r, 1.0, -1, cv2.LINE_AA)
    band = f.to_img([[-0.78, base_v], [0.78, base_v]])
    cv2.line(layer, tuple(band[0]), tuple(band[1]), (255, 255, 255), max(3, int(f.d * 0.12)), cv2.LINE_AA)
    cv2.line(alpha, tuple(band[0]), tuple(band[1]), 1.0, max(3, int(f.d * 0.12)), cv2.LINE_AA)
    return _blend(img, layer, alpha)


def _moustache(img, f: FaceFrame):
    nose_v = f.to_local(2)[1]     # base of nose
    lip_v = f.to_local(0)[1]      # top of upper lip
    cv_ = (nose_v + lip_v) / 2 + 0.02
    mw = abs(f.to_local(291)[0] - f.to_local(61)[0]) / 2 + 0.12
    hh = max(0.07, (lip_v - nose_v) * 0.55)
    half = np.array([[0.0, -hh * 0.6], [0.25, -hh], [0.55, -hh * 0.9], [0.85, -hh * 0.2],
                     [1.05, -hh * 0.9], [1.12, -hh * 1.6], [1.18, -hh * 0.6], [1.0, hh * 0.5],
                     [0.7, hh * 0.9], [0.35, hh], [0.0, hh * 0.4]])
    right = half[::-1] * [-1, 1]
    shape = np.vstack([half, right]) * [mw, 1] + [0, cv_]
    poly = f.to_img(shape)
    layer = img.copy()
    alpha = np.zeros(img.shape[:2], np.float32)
    cv2.fillPoly(layer, [poly], (30, 35, 50), cv2.LINE_AA)
    cv2.fillPoly(alpha, [poly], 0.95, cv2.LINE_AA)
    return _blend(img, layer, alpha)


def _cat_face(img, f: FaceFrame):
    layer = img.copy()
    alpha = np.zeros(img.shape[:2], np.float32)
    nose = f.to_local(4)
    tri = f.to_img([[nose[0] - 0.13, nose[1] - 0.08], [nose[0] + 0.13, nose[1] - 0.08], [nose[0], nose[1] + 0.06]])
    cv2.fillPoly(layer, [tri], (160, 110, 255), cv2.LINE_AA)
    cv2.fillPoly(alpha, [tri], 1.0, cv2.LINE_AA)
    th = max(2, int(f.d * 0.045))
    for side in (-1, 1):
        for dv in (-0.08, 0.04, 0.16):
            w = f.to_img([[nose[0] + side * 0.3, nose[1] + 0.12 + dv * 0.3],
                          [nose[0] + side * 1.05, nose[1] + 0.02 + dv]])
            cv2.line(layer, tuple(w[0]), tuple(w[1]), (40, 40, 40), th + 2, cv2.LINE_AA)
            cv2.line(layer, tuple(w[0]), tuple(w[1]), (250, 250, 250), th, cv2.LINE_AA)
            cv2.line(alpha, tuple(w[0]), tuple(w[1]), 1.0, th + 2, cv2.LINE_AA)
    top_v = f.to_local(10)[1]
    for side in (-1, 1):
        outer = f.to_img([[side * 0.35, top_v + 0.1], [side * 0.95, top_v - 0.75], [side * 1.05, top_v + 0.25]])
        inner = f.to_img([[side * 0.5, top_v + 0.05], [side * 0.9, top_v - 0.5], [side * 0.95, top_v + 0.15]])
        cv2.fillPoly(layer, [outer], (40, 40, 40), cv2.LINE_AA)
        cv2.fillPoly(alpha, [outer], 1.0, cv2.LINE_AA)
        cv2.fillPoly(layer, [inner], (170, 130, 255), cv2.LINE_AA)
    return _blend(img, layer, alpha)


AR_OVERLAYS = {
    "Sunglasses": _sunglasses,
    "Party Hat": _party_hat,
    "Moustache": _moustache,
    "Cat Face": _cat_face,
}


def draw_ar_overlay(image: np.ndarray, faces_landmarks, overlays) -> np.ndarray:
    """overlays: iterable of names from AR_OVERLAYS, applied to every face."""
    out = image
    for lm in faces_landmarks:
        f = FaceFrame(lm)
        for name in overlays:
            out = AR_OVERLAYS[name](out, f)
    return out


# ----------------------------------------------------------------------
# Privacy / anonymisation
# ----------------------------------------------------------------------
def anonymize_faces(image: np.ndarray, faces, mode: str = "Blur", pad: float = 0.25) -> np.ndarray:
    """faces: output of FaceDetector.detect(). mode: Blur | Pixelate | Eye bar | Solid."""
    out = image.copy()
    H, W = image.shape[:2]
    for face in faces:
        x, y, w, h = face["bbox"]
        if mode == "Eye bar" and len(face["keypoints"]) >= 2:
            (x1, y1), (x2, y2) = face["keypoints"][0], face["keypoints"][1]
            c = ((x1 + x2) / 2, (y1 + y2) / 2)
            length = np.hypot(x2 - x1, y2 - y1) * 2.0
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            box = cv2.boxPoints((c, (length, length * 0.32), ang)).astype(np.int32)
            cv2.fillPoly(out, [box], (0, 0, 0), cv2.LINE_AA)
            continue
        px, py = int(w * pad), int(h * pad)
        x0, y0 = max(x - px, 0), max(y - py, 0)
        x1, y1 = min(x + w + px, W), min(y + h + py, H)
        roi = out[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        if mode == "Pixelate":
            bs = max(6, (x1 - x0) // 12)
            small = cv2.resize(roi, (max(1, roi.shape[1] // bs), max(1, roi.shape[0] // bs)),
                               interpolation=cv2.INTER_LINEAR)
            effect = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        elif mode == "Solid":
            effect = np.full_like(roi, (30, 30, 30))
        else:
            k = (max(roi.shape[:2]) // 3) | 1
            effect = cv2.GaussianBlur(roi, (k, k), 0)
            effect = cv2.GaussianBlur(effect, (k, k), 0)
        m = np.zeros(roi.shape[:2], np.float32)
        cv2.ellipse(m, (roi.shape[1] // 2, roi.shape[0] // 2),
                    (roi.shape[1] // 2, roi.shape[0] // 2), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), max(1, roi.shape[1] // 30))
        out[y0:y1, x0:x1] = _blend(roi, effect, m)
    return out


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------
def head_pose(lm: np.ndarray, image_shape=None):
    """Head orientation from MediaPipe's 3-D landmarks (no generic head model needed).

    An orthonormal head frame is built with Gram-Schmidt:
        x_h = eye-corner 33 -> eye-corner 263      (across the face)
        y_h = forehead 10 -> chin 152, minus its x_h component (down the face)
        z_h = x_h cross y_h                          (into the screen)
    R = [x_h y_h z_h] is the head-to-camera rotation; Euler angles follow from it.

    Sign convention:
        yaw   > 0 : face turned towards image-right
        pitch > 0 : face tilted up
        roll  > 0 : head tilted clockwise in the image
    """
    p = lm[:, :3].astype(np.float64)
    xa = p[263] - p[33]
    xa /= np.linalg.norm(xa) + 1e-9
    ya = p[152] - p[10]
    ya -= (ya @ xa) * xa
    ya /= np.linalg.norm(ya) + 1e-9
    za = np.cross(xa, ya)
    yaw = np.degrees(np.arctan2(-za[0], za[2]))
    pitch = np.degrees(np.arctan2(za[1], np.hypot(za[0], za[2])))
    roll = np.degrees(np.arctan2(xa[1], xa[0]))
    return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll),
            "R": np.stack([xa, ya, za], axis=1)}


def describe_pose(p, thresh: float = 15.0) -> str:
    parts = []
    if p["yaw"] > thresh:
        parts.append("turned right")
    elif p["yaw"] < -thresh:
        parts.append("turned left")
    if p["pitch"] > thresh:
        parts.append("looking up")
    elif p["pitch"] < -thresh:
        parts.append("looking down")
    if abs(p["roll"]) > thresh:
        parts.append("head tilted")
    return ", ".join(parts) if parts else "facing camera"


def draw_head_pose(image: np.ndarray, lm: np.ndarray, pose) -> np.ndarray:
    """Draw the head frame at the nose tip: X red, Y green, Z (towards camera) blue."""
    out = image
    nose = lm[1, :2]
    L = 1.3 * float(np.linalg.norm(lm[263, :2] - lm[33, :2]))
    R = pose["R"]
    th = max(2, image.shape[1] // 300)
    for axis, col in ((R[:, 0], (60, 60, 255)), (R[:, 1], (60, 220, 60)), (-R[:, 2], (255, 120, 40))):
        end = nose + L * axis[:2]
        cv2.arrowedLine(out, tuple(int(v) for v in nose), tuple(int(v) for v in end), col, th,
                        cv2.LINE_AA, tipLength=0.15)
    return out


def eye_aspect_ratio(lm: np.ndarray, idx) -> float:
    p = lm[idx, :2]
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3]) + 1e-6
    return float((a + b) / (2.0 * c))


def eye_state(lm: np.ndarray, threshold: float = 0.20):
    r, l = eye_aspect_ratio(lm, EAR_RIGHT), eye_aspect_ratio(lm, EAR_LEFT)
    ear = (r + l) / 2
    return {"ear_right": r, "ear_left": l, "ear": ear, "closed": ear < threshold}


def mouth_aspect_ratio(lm: np.ndarray) -> float:
    v = np.linalg.norm(lm[13, :2] - lm[14, :2])
    h = np.linalg.norm(lm[61, :2] - lm[291, :2]) + 1e-6
    return float(v / h)


def face_geometry(lm: np.ndarray):
    """Simple proportions derived from landmarks."""
    width = np.linalg.norm(lm[234, :2] - lm[454, :2])
    height = np.linalg.norm(lm[10, :2] - lm[152, :2])
    f = FaceFrame(lm)
    # symmetry: compare distances of mirrored landmark pairs from the mid-line
    pairs = [(33, 263), (61, 291), (234, 454), (70, 300), (172, 397)]
    mid = lm[[10, 152], :2]
    axis = (mid[1] - mid[0]) / (np.linalg.norm(mid[1] - mid[0]) + 1e-6)
    normal = np.array([-axis[1], axis[0]])
    diffs = []
    for a, b in pairs:
        da = abs((lm[a, :2] - mid[0]) @ normal)
        db = abs((lm[b, :2] - mid[0]) @ normal)
        diffs.append(abs(da - db) / (max(da, db) + 1e-6))
    return {"face_width_px": float(width), "face_height_px": float(height),
            "height_width_ratio": float(height / (width + 1e-6)),
            "interocular_px": f.d, "roll_deg": f.roll_deg,
            "symmetry_pct": float(100 * (1 - np.mean(diffs)))}


def draw_face_info(image, faces_info):
    """faces_info: list of (face_number, pose, eyes, mar). Draws a read-out panel top-left."""
    out = image
    lines = []
    for n, pose, eyes, mar in faces_info:
        lines += [f"Face {n}: {describe_pose(pose)}",
                  f"  Yaw {pose['yaw']:+.0f}  Pitch {pose['pitch']:+.0f}  Roll {pose['roll']:+.0f} deg",
                  f"  Eyes {'CLOSED' if eyes['closed'] else 'open'} (EAR {eyes['ear']:.2f})"
                  f"  Mouth {'open' if mar > 0.35 else 'closed'} (MAR {mar:.2f})"]
    if not lines:
        return out
    scale = max(0.45, image.shape[1] / 1400)
    lh = int(26 * scale / 0.45 * 0.75)
    (tw, _), _ = cv2.getTextSize(max(lines, key=len), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    panel = out.copy()
    cv2.rectangle(panel, (8, 8), (20 + tw, 16 + lh * len(lines)), (20, 20, 20), -1)
    out[:] = cv2.addWeighted(panel, 0.65, out, 0.35, 0)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (14, 8 + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------
# Beauty
# ----------------------------------------------------------------------
def _poly_mask(shape, lm, idx, value=1.0):
    m = np.zeros(shape[:2], np.float32)
    cv2.fillPoly(m, [lm[idx, :2].astype(np.int32)], value, cv2.LINE_AA)
    return m


def skin_mask(shape, lm: np.ndarray, feather: int = 0) -> np.ndarray:
    f = FaceFrame(lm)
    m = _poly_mask(shape, lm, FACE_OVAL)
    holes = np.zeros(shape[:2], np.float32)
    for idx in (LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS_OUTER):
        holes = np.maximum(holes, _poly_mask(shape, lm, idx))
    k = max(3, int(f.d * 0.12)) | 1
    holes = cv2.dilate(holes, np.ones((k, k), np.uint8))
    m = np.clip(m - holes, 0, 1)
    feather = feather or max(3, int(f.d * 0.15))
    return cv2.GaussianBlur(m, (0, 0), feather / 2)


def beautify(image: np.ndarray, faces_landmarks, smooth: float = 0.6, brighten: int = 8) -> np.ndarray:
    """Edge-preserving skin smoothing applied only inside the skin mask."""
    if smooth <= 0 and brighten == 0:
        return image
    out = image
    for lm in faces_landmarks:
        d = int(9 + 6 * smooth)
        smoothed = cv2.bilateralFilter(out, d, 40 + 50 * smooth, 40 + 50 * smooth)
        if brighten:
            smoothed = cv2.convertScaleAbs(smoothed, alpha=1.0, beta=brighten)
        m = skin_mask(out.shape, lm) * min(1.0, smooth + (0.3 if brighten else 0))
        out = _blend(out, smoothed, m)
    return out


def tint_lips(image: np.ndarray, faces_landmarks, hex_color: str = "#c8324b", strength: float = 0.5) -> np.ndarray:
    if strength <= 0:
        return image
    out = image
    target = _hex_hsv(hex_color)
    for lm in faces_landmarks:
        m = np.clip(_poly_mask(out.shape, lm, LIPS_OUTER) - _poly_mask(out.shape, lm, LIPS_INNER), 0, 1)
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, FaceFrame(lm).d * 0.02))
        out = _blend(out, _recolor(out, target), m * strength)
    return out


def _iris(lm, eye_idx, iris_idx):
    eye_poly = lm[eye_idx, :2]
    center = lm[iris_idx[0], :2]
    radius = float(np.mean(np.linalg.norm(lm[iris_idx[1:], :2] - center, axis=1)))
    eye_w = np.linalg.norm(eye_poly.max(0) - eye_poly.min(0))
    inside = cv2.pointPolygonTest(eye_poly.astype(np.float32), tuple(map(float, center)), False) >= 0
    if not inside or not (0.1 * eye_w < radius < 0.45 * eye_w):
        center = eye_poly.mean(axis=0)          # fallback when iris points are unavailable
        radius = 0.22 * float(np.ptp(eye_poly[:, 0]))
    return center, radius


def tint_eyes(image: np.ndarray, faces_landmarks, hex_color: str = "#2a9df4", strength: float = 0.5) -> np.ndarray:
    if strength <= 0:
        return image
    out = image
    target = _hex_hsv(hex_color)
    for lm in faces_landmarks:
        m = np.zeros(out.shape[:2], np.float32)
        for eye_idx, iris_idx in ((LEFT_EYE, LEFT_IRIS), (RIGHT_EYE, RIGHT_IRIS)):
            c, r = _iris(lm, eye_idx, iris_idx)
            circ = np.zeros_like(m)
            cv2.circle(circ, (int(c[0]), int(c[1])), max(1, int(r)), 1.0, -1, cv2.LINE_AA)
            m = np.maximum(m, circ * _poly_mask(out.shape, lm, eye_idx))
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        out = _blend(out, _recolor(out, target), m * strength)
    return out


def _hex_hsv(hex_color):
    h = hex_color.lstrip("#")
    rgb = np.uint8([[[int(h[i:i + 2], 16) for i in (0, 2, 4)]]])
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0]


def _recolor(image, target_hsv):
    """Replace hue & saturation with target's but keep the original brightness (texture)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = target_hsv[0]
    hsv[..., 1] = np.maximum(hsv[..., 1], target_hsv[1] * 0.8).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
