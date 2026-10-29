"""
face_utils.py
----------------------------------------------------------------------
Face & person analysis built on Google's MediaPipe Tasks API, plus a
classic Haar-cascade detector kept ONLY as a baseline for comparison.

    FaceDetector       BlazeFace CNN  -> bounding box + 6 keypoints
    HaarFaceDetector   Viola-Jones    -> bounding box (baseline)
    FaceMeshDetector   FaceLandmarker -> 478 dense 3-D landmarks
    PersonSegmenter    Selfie model   -> per-pixel person probability

All public functions take/return OpenCV-style BGR uint8 images.
Landmarks are returned as NumPy arrays of PIXEL coordinates, shape
(478, 3) -> columns x, y, z (z is relative depth, same scale as x).
----------------------------------------------------------------------
"""

import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from download_models import HAAR_FILE, ensure_model

# ----------------------------------------------------------------------
# Landmark index groups (MediaPipe 478-point topology)
# ----------------------------------------------------------------------
NUM_LANDMARKS = 478  # 468 face-surface points + 10 iris points

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314,
              17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317,
              14, 87, 178, 88, 95]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,
            384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160,
             161, 246]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]

# Six points per eye for the Eye Aspect Ratio (Soukupova & Cech, 2016)
EAR_RIGHT = [33, 160, 158, 133, 153, 144]
EAR_LEFT = [362, 385, 387, 263, 373, 380]



def _to_mp_image(image_bgr: np.ndarray) -> mp.Image:
    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


# ----------------------------------------------------------------------
# BlazeFace detector
# ----------------------------------------------------------------------
class FaceDetector:
    """Deep-learning face detection (BlazeFace): box + 6 keypoints + score."""

    KEYPOINT_LABELS = ["Right Eye", "Left Eye", "Nose Tip", "Mouth", "Right Ear", "Left Ear"]
    KEYPOINT_COLORS = [(255, 120, 0), (255, 120, 0), (0, 230, 255),
                       (60, 60, 255), (255, 0, 200), (255, 0, 200)]

    def __init__(self, min_confidence: float = 0.5, multiscale: bool = True,
                 tile_levels=(2, 3, 4), tile_min_score: float = 0.6):
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("blaze_face_short_range.tflite")),
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self.multiscale = multiscale
        self.tile_levels = tile_levels
        self.tile_min_score = tile_min_score   # tiles see less context -> demand more confidence

    def _detect_once(self, image_bgr, ox=0, oy=0):
        h, w = image_bgr.shape[:2]
        result = self._detector.detect(_to_mp_image(image_bgr))
        faces = []
        for det in result.detections:
            bb = det.bounding_box
            x, y = max(bb.origin_x, 0), max(bb.origin_y, 0)
            bw, bh = min(bb.width, w - x), min(bb.height, h - y)
            if bw <= 0 or bh <= 0:
                continue
            kps = [(int(k.x * w) + ox, int(k.y * h) + oy) for k in (det.keypoints or [])]
            score = det.categories[0].score if det.categories else 0.0
            faces.append({"bbox": (x + ox, y + oy, bw, bh), "keypoints": kps, "score": float(score)})
        return faces

    def detect(self, image_bgr: np.ndarray, multiscale=None):
        """List of {bbox: (x, y, w, h), keypoints: [(x, y)...], score}.

        BlazeFace *short-range* is designed for selfie distance: it resizes the whole
        image to 128x128, so faces smaller than ~10 % of the frame are missed. With
        multiscale=True the image is also split into overlapping 2x2, 3x3 and 4x4 tiles
        (25 % overlap); each tile is detected separately, tile detections below
        tile_min_score are discarded, boxes are mapped back to full-image coordinates
        and duplicates are merged with non-maximum suppression.
        """
        faces = self._detect_once(image_bgr)
        if not (self.multiscale if multiscale is None else multiscale):
            return faces
        H, W = image_bgr.shape[:2]
        for n in self.tile_levels:
            tw, th = int(W / n * 1.25), int(H / n * 1.25)
            for i in range(n):
                for j in range(n):
                    x0 = min(int(j * W / n - (tw - W / n) / 2), W - tw) if n > 1 else 0
                    y0 = min(int(i * H / n - (th - H / n) / 2), H - th) if n > 1 else 0
                    x0, y0 = max(x0, 0), max(y0, 0)
                    tile = image_bgr[y0:y0 + th, x0:x0 + tw]
                    faces += [f for f in self._detect_once(tile, x0, y0) if f["score"] >= self.tile_min_score]
        return nms(faces, iou_thresh=0.3)

    def draw_detections(self, image_bgr: np.ndarray, faces=None, label_keypoints=True) -> np.ndarray:
        out = image_bgr.copy()
        faces = self.detect(image_bgr) if faces is None else faces
        thick = max(2, image_bgr.shape[1] // 400)
        for f in faces:
            x, y, w, h = f["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 120), thick)
            _label(out, f"Face {f['score'] * 100:.0f}%", (x, y), (0, 220, 120))
            for i, (kx, ky) in enumerate(f["keypoints"]):
                col = self.KEYPOINT_COLORS[i % len(self.KEYPOINT_COLORS)]
                cv2.circle(out, (kx, ky), thick + 3, col, -1)
                cv2.circle(out, (kx, ky), thick + 3, (255, 255, 255), 1)
                if label_keypoints and w > 120:
                    cv2.putText(out, self.KEYPOINT_LABELS[i], (kx + 6, ky - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def close(self):
        self._detector.close()


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


def nms(faces, iou_thresh=0.3):
    """Greedy non-maximum suppression. Also drops a box mostly contained in a kept one."""
    kept = []
    for f in sorted(faces, key=lambda d: -d["score"]):
        ok = True
        for k in kept:
            x, y, w, h = f["bbox"]
            kx, ky, kw, kh = k["bbox"]
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            contained = ix * iy / float(min(w * h, kw * kh) + 1e-9)
            if iou(f["bbox"], k["bbox"]) > iou_thresh or contained > 0.6:
                ok = False
                break
        if ok:
            kept.append(f)
    return kept


# ----------------------------------------------------------------------
# Haar cascade (baseline only)
# ----------------------------------------------------------------------
class HaarFaceDetector:
    """Classic Viola-Jones Haar cascade - used only as a comparison baseline."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        path = ensure_model(HAAR_FILE)
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, image_bgr: np.ndarray):
        gray = cv2.equalizeHist(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
        min_side = max(24, min(image_bgr.shape[:2]) // 20)
        boxes = self._cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(min_side, min_side))
        return [{"bbox": tuple(int(v) for v in b), "keypoints": [], "score": None} for b in boxes]

    def draw_detections(self, image_bgr: np.ndarray, faces=None) -> np.ndarray:
        out = image_bgr.copy()
        faces = self.detect(image_bgr) if faces is None else faces
        thick = max(2, image_bgr.shape[1] // 400)
        for f in faces:
            x, y, w, h = f["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 140, 255), thick)
            _label(out, "Haar face", (x, y), (0, 140, 255))
        return out


def timed(fn, *args, repeats: int = 1, **kwargs):
    """Run fn and return (result, average milliseconds)."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        res = fn(*args, **kwargs)
    return res, (time.perf_counter() - t0) * 1000 / repeats


# ----------------------------------------------------------------------
# Dense 478-point face mesh
# ----------------------------------------------------------------------
class FaceMeshDetector:
    """478-point facial landmark 'breakdown' using MediaPipe FaceLandmarker."""

    def __init__(self, max_faces: int = 5, min_confidence: float = 0.5):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("face_landmarker.task")),
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        self._conn = vision.FaceLandmarksConnections
        self._styles = vision.drawing_styles
        self._drawing = vision.drawing_utils

    def _raw(self, image_bgr):
        return self._detector.detect(_to_mp_image(image_bgr))

    def _landmarks_once(self, image_bgr, ox=0.0, oy=0.0, scale=1.0):
        h, w = image_bgr.shape[:2]
        out = []
        for lms in self._raw(image_bgr).face_landmarks:
            arr = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float32)
            arr /= scale
            arr[:, 0] += ox
            arr[:, 1] += oy
            out.append(arr)
        return out

    def landmarks(self, image_bgr: np.ndarray, boxes=None):
        """List (one per face) of (478, 3) float32 arrays in pixel units.

        Two-stage refinement: FaceLandmarker's built-in detector is short-range, so
        small faces are missed. If `boxes` (e.g. from the multi-scale FaceDetector) are
        given, every box not already covered by a mesh is cropped (2x the box size),
        enlarged to >= 384 px, landmarked, and the points are mapped back.
        """
        faces = self._landmarks_once(image_bgr)
        if not boxes:
            return faces
        H, W = image_bgr.shape[:2]
        for (x, y, w, h) in [b["bbox"] if isinstance(b, dict) else b for b in boxes]:
            cx, cy = x + w / 2, y + h / 2
            if any(f[:, 0].min() <= cx <= f[:, 0].max() and f[:, 1].min() <= cy <= f[:, 1].max() for f in faces):
                continue
            side = 2.0 * max(w, h)
            x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
            x1, y1 = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
            crop = image_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            scale = max(1.0, 384.0 / max(crop.shape[:2]))
            if scale > 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            found = self._landmarks_once(crop, x0, y0, scale)
            if found:
                # keep the mesh whose centre is closest to the box centre
                found.sort(key=lambda f: np.hypot(f[:, 0].mean() - cx, f[:, 1].mean() - cy))
                faces.append(found[0])
        return faces

    def draw_mesh(self, image_bgr: np.ndarray, style: str = "tesselation", faces=None) -> np.ndarray:
        """style: 'tesselation' | 'contours' | 'irises' | 'points' | 'all'."""
        out = image_bgr.copy()
        faces = self.landmarks(image_bgr) if faces is None else faces
        groups = []
        if style in ("tesselation", "all"):
            groups.append((self._conn.FACE_LANDMARKS_TESSELATION, (192, 192, 192), 1))
        if style in ("contours", "all"):
            groups.append((self._conn.FACE_LANDMARKS_CONTOURS, (80, 230, 80), 2))
        if style in ("irises", "all"):
            groups.append((list(self._conn.FACE_LANDMARKS_LEFT_IRIS) + list(self._conn.FACE_LANDMARKS_RIGHT_IRIS),
                           (255, 200, 30), 2))
        for lm in faces:
            pts = lm[:, :2].astype(np.int32)
            size = float(np.ptp(pts[:, 0]))
            for conns, col, th in groups:
                t = 1 if size < 200 else th
                for c in conns:
                    a, b = (c.start, c.end) if hasattr(c, "start") else c
                    cv2.line(out, tuple(pts[a]), tuple(pts[b]), col, t, cv2.LINE_AA)
            if style == "points":
                for p in pts:
                    cv2.circle(out, tuple(p), 1, (0, 255, 200), -1)
        return out

    def close(self):
        self._detector.close()


# ----------------------------------------------------------------------
# Person segmentation
# ----------------------------------------------------------------------
class PersonSegmenter:
    """Person / background segmentation for portrait-mode effects."""

    def __init__(self):
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("selfie_segmenter.tflite")),
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def get_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """float32 mask in [0, 1], same H x W as input. 1 = person."""
        result = self._segmenter.segment(_to_mp_image(image_bgr))
        masks = result.confidence_masks
        # Official selfie_segmenter returns [background, person]; older single-output
        # models return only the person channel. Take the LAST mask either way.
        mask = np.array(masks[-1].numpy_view(), dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        h, w = image_bgr.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h))
        return np.clip(mask, 0, 1)

    def close(self):
        self._segmenter.close()


def _label(img, text, origin, color):
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y0 = max(y - th - 8, 0)
    cv2.rectangle(img, (x, y0), (x + tw + 8, y0 + th + 8), color, -1)
    cv2.putText(img, text, (x + 4, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (20, 20, 20), 1, cv2.LINE_AA)
