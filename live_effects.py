"""
live_effects.py
----------------------------------------------------------------------
Per-frame processing for real-time video, shared by the in-app Live
Camera tab (streamlit-webrtc) and the stand-alone desktop window
(live_camera.py).
----------------------------------------------------------------------
"""

import time
from collections import deque

import cv2
import numpy as np

import face_effects as fx
import filters

LIVE_MODES = [
    "AR: Sunglasses",
    "AR: Party Hat + Sunglasses",
    "AR: Moustache",
    "AR: Cat Face",
    "Head pose + blink counter",
    "Face mesh (478 points)",
    "Face detection (BlazeFace)",
    "Privacy: blur faces",
    "Privacy: pixelate faces",
    "Portrait mode (background blur)",
    "Background: studio colour",
    "Beauty (skin smoothing)",
    "Cartoon",
    "Pencil Sketch",
    "Original",
]

_AR = {"AR: Sunglasses": ["Sunglasses"], "AR: Party Hat + Sunglasses": ["Party Hat", "Sunglasses"],
       "AR: Moustache": ["Moustache"], "AR: Cat Face": ["Cat Face"]}


class BlinkCounter:
    """Counts blinks: EAR below threshold for >= min_frames, then back above."""

    def __init__(self, threshold: float = 0.20, min_frames: int = 2):
        self.threshold, self.min_frames = threshold, min_frames
        self.closed_frames = 0
        self.blinks = 0

    def update(self, ear: float) -> bool:
        if ear < self.threshold:
            self.closed_frames += 1
            return True
        if self.closed_frames >= self.min_frames:
            self.blinks += 1
        self.closed_frames = 0
        return False


class LiveProcessor:
    """Holds models + state for one video stream."""

    def __init__(self, models):
        self.models = models            # pipeline.Models-like object
        self.mode = LIVE_MODES[0]
        self.mirror = True
        self.show_fps = True
        self.blinks = BlinkCounter()
        self._times = deque(maxlen=30)
        self._last_mode = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.mode != self._last_mode:      # restart the FPS average when the effect changes
            self._times.clear()
            self._last_mode = self.mode
        t0 = time.perf_counter()
        if self.mirror:
            frame = cv2.flip(frame, 1)
        out = self._apply(frame)
        self._times.append(time.perf_counter() - t0)
        if self.show_fps:
            fps = len(self._times) / max(sum(self._times), 1e-6)
            _hud(out, f"{self.mode}  |  {fps:4.1f} FPS  |  {1000 * np.mean(self._times):.0f} ms/frame")
        return out

    def _apply(self, img):
        m, mode = self.models, self.mode
        if mode in _AR:
            return fx.draw_ar_overlay(img, m.mesh.landmarks(img), _AR[mode])
        if mode == "Head pose + blink counter":
            lms = m.mesh.landmarks(img)
            out, info = img.copy(), []
            for i, lm in enumerate(lms, 1):
                pose = fx.head_pose(lm)
                eyes = fx.eye_state(lm, self.blinks.threshold)
                if i == 1:
                    self.blinks.update(eyes["ear"])
                fx.draw_head_pose(out, lm, pose)
                info.append((i, pose, eyes, fx.mouth_aspect_ratio(lm)))
            fx.draw_face_info(out, info)
            _hud(out, f"Blinks: {self.blinks.blinks}", row=1)
            return out
        if mode == "Face mesh (478 points)":
            return m.mesh.draw_mesh(img, "all")
        if mode == "Face detection (BlazeFace)":
            return m.detector.draw_detections(img, m.detector.detect(img, multiscale=False))
        if mode.startswith("Privacy"):
            return fx.anonymize_faces(img, m.detector.detect(img, multiscale=False), "Blur" if "blur" in mode else "Pixelate")
        if mode == "Portrait mode (background blur)":
            return filters.blur_background(img, m.segmenter.get_mask(img), 18, 8)
        if mode == "Background: studio colour":
            return filters.replace_background_color(img, m.segmenter.get_mask(img), (58, 42, 30), 6)
        if mode == "Beauty (skin smoothing)":
            return fx.beautify(img, m.mesh.landmarks(img), 0.6, 6)
        if mode == "Cartoon":
            return filters.cartoonize(img)
        if mode == "Pencil Sketch":
            return filters.pencil_sketch(img)
        return img


def _hud(img, text, row=0):
    y = img.shape[0] - 12 - row * 28
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (6, y - th - 8), (16 + tw, y + 6), (20, 20, 20), -1)
    cv2.putText(img, text, (11, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)
