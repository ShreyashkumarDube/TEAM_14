"""
pipeline.py
----------------------------------------------------------------------
The ordered, stackable processing pipeline shared by the Streamlit
app, batch mode, the live camera and the benchmark script.

    Input -> 1 Enhance -> 2 Beauty -> 3 Background -> 4 Style
          -> 5 Finishing (vignette, grain) -> 6 AR overlay -> 7 Privacy -> Output

Face landmarks, the segmentation mask and face boxes are computed ONCE
from the input image (they do not change, because no stage moves
pixels geometrically) and are supplied lazily through getter callables
so that a stage which is switched off costs nothing.
----------------------------------------------------------------------
"""

import time
from dataclasses import dataclass, field

import face_effects as fx
import filters

DEFAULTS = {
    # 1 Enhance
    "auto_enhance": False, "brightness": 0, "contrast": 0, "saturation": 0,
    "gamma": 1.0, "sharpness": 0.0, "warmth": 0,
    # 2 Beauty
    "beauty_smooth": 0.0, "beauty_brighten": 0, "lip_strength": 0.0, "lip_color": "#c8324b",
    "eye_strength": 0.0, "eye_color": "#2a9df4",
    # 3 Background
    "bg_mode": "None", "bg_strength": 15, "bg_feather": 10, "bg_color": "#1e2a3a",
    # 4 Style
    "style": "None",
    # 5 Finishing
    "vignette": 0.0, "grain": 0,
    # 6 AR
    "ar_overlays": [],
    # 7 Privacy
    "privacy": "Off",
}

BG_MODES = ["None", "Blur (portrait mode)", "Solid colour", "Custom image", "Colour splash"]
PRIVACY_MODES = ["Off", "Blur", "Pixelate", "Eye bar", "Solid"]

PRESETS = {
    "Portrait Pro": {"auto_enhance": True, "beauty_smooth": 0.5, "beauty_brighten": 6,
                     "bg_mode": "Blur (portrait mode)", "bg_strength": 18, "vignette": 0.6, "sharpness": 0.4},
    "Vintage": {"contrast": -10, "saturation": -25, "warmth": 25, "style": "Sepia",
                "vignette": 1.6, "grain": 14},
    "Noir": {"style": "Noir (high-contrast B&W)", "vignette": 1.4, "grain": 8},
    "Cartoon Pop": {"saturation": 35, "style": "Cartoon", "ar_overlays": ["Sunglasses"]},
    "Backdrop": {"auto_enhance": True, "bg_mode": "Solid colour", "bg_color": "#1e2a3a",
                        "bg_feather": 8, "beauty_smooth": 0.4},
    "Party Mode": {"saturation": 30, "warmth": 10, "ar_overlays": ["Party Hat", "Sunglasses"]},
    "Privacy Safe": {"privacy": "Pixelate"},
}


def settings_from(overrides: dict | None = None) -> dict:
    s = dict(DEFAULTS)
    s["ar_overlays"] = list(DEFAULTS["ar_overlays"])
    if overrides:
        s.update(overrides)
    return s


@dataclass
class PipelineResult:
    image: object
    timings: list = field(default_factory=list)   # [(stage, ms)]
    notes: list = field(default_factory=list)

    @property
    def total_ms(self):
        return sum(t for _, t in self.timings)


def needs_landmarks(s):
    return (s["beauty_smooth"] > 0 or s["beauty_brighten"] or s["lip_strength"] > 0
            or s["eye_strength"] > 0 or bool(s["ar_overlays"]))


def run_pipeline(img, s, get_landmarks=None, get_mask=None, get_faces=None, bg_image=None) -> PipelineResult:
    res = PipelineResult(image=img)
    out = img

    def stage(name, fn):
        nonlocal out
        t0 = time.perf_counter()
        out = fn(out)
        res.timings.append((name, (time.perf_counter() - t0) * 1000))

    # 1 Enhance
    if (s["auto_enhance"] or s["brightness"] or s["contrast"] or s["saturation"]
            or abs(s["gamma"] - 1) > 1e-3 or s["sharpness"] or s["warmth"]):
        def enhance(x):
            if s["auto_enhance"]:
                x = filters.auto_enhance(x)
            x = filters.adjust_brightness(x, s["brightness"])
            x = filters.adjust_contrast(x, s["contrast"])
            x = filters.adjust_saturation(x, s["saturation"])
            x = filters.adjust_gamma(x, s["gamma"])
            x = filters.adjust_sharpness(x, s["sharpness"])
            return filters.adjust_warmth(x, s["warmth"])
        stage("1 Enhance", enhance)

    lms = None
    if needs_landmarks(s) and get_landmarks is not None:
        t0 = time.perf_counter()
        lms = get_landmarks()
        res.timings.append(("Face mesh (478 pts)", (time.perf_counter() - t0) * 1000))
        if not lms:
            res.notes.append("No face found - face effects (beauty / AR) were skipped.")

    # 2 Beauty
    if lms and (s["beauty_smooth"] > 0 or s["beauty_brighten"] or s["lip_strength"] > 0 or s["eye_strength"] > 0):
        def beauty(x):
            x = fx.beautify(x, lms, s["beauty_smooth"], s["beauty_brighten"])
            x = fx.tint_lips(x, lms, s["lip_color"], s["lip_strength"])
            return fx.tint_eyes(x, lms, s["eye_color"], s["eye_strength"])
        stage("2 Beauty", beauty)

    # 3 Background
    mode = s["bg_mode"]
    if mode != "None" and get_mask is not None:
        if mode == "Custom image" and bg_image is None:
            res.notes.append("Upload a background image to use 'Custom image' mode.")
        else:
            t0 = time.perf_counter()
            mask = get_mask()
            res.timings.append(("Segmentation", (time.perf_counter() - t0) * 1000))

            def background(x):
                if mode.startswith("Blur"):
                    return filters.blur_background(x, mask, s["bg_strength"], s["bg_feather"])
                if mode == "Solid colour":
                    return filters.replace_background_color(x, mask, filters.hex_to_bgr(s["bg_color"]), s["bg_feather"])
                if mode == "Custom image":
                    return filters.replace_background_image(x, mask, bg_image, s["bg_feather"])
                return filters.greyscale_background(x, mask, s["bg_feather"])
            stage("3 Background", background)

    # 4 Style
    fn = filters.STYLES.get(s["style"])
    if fn is not None:
        stage(f"4 Style: {s['style']}", fn)

    # 5 Finishing
    if s["vignette"] > 0 or s["grain"] > 0:
        stage("5 Vignette / grain",
              lambda x: filters.add_film_grain(filters.apply_vignette(x, s["vignette"]), s["grain"]))

    # 6 AR overlay
    if lms and s["ar_overlays"]:
        stage("6 AR overlay", lambda x: fx.draw_ar_overlay(x, lms, s["ar_overlays"]))

    # 7 Privacy
    if s["privacy"] != "Off" and get_faces is not None:
        t0 = time.perf_counter()
        faces = get_faces()
        res.timings.append(("Face detection", (time.perf_counter() - t0) * 1000))
        stage(f"7 Privacy: {s['privacy']}", lambda x: fx.anonymize_faces(x, faces, s["privacy"]))
        if not faces:
            res.notes.append("Privacy mode: no faces detected to anonymise.")

    res.image = out
    return res


class Models:
    """Lazily-created model holders for scripts that run outside Streamlit."""

    def __init__(self):
        self._det = self._mesh = self._seg = self._haar = None

    @property
    def detector(self):
        if self._det is None:
            from face_utils import FaceDetector
            self._det = FaceDetector()
        return self._det

    @property
    def mesh(self):
        if self._mesh is None:
            from face_utils import FaceMeshDetector
            self._mesh = FaceMeshDetector()
        return self._mesh

    @property
    def segmenter(self):
        if self._seg is None:
            from face_utils import PersonSegmenter
            self._seg = PersonSegmenter()
        return self._seg

    @property
    def haar(self):
        if self._haar is None:
            from face_utils import HaarFaceDetector
            self._haar = HaarFaceDetector()
        return self._haar

    def run(self, img, s, bg_image=None):
        return run_pipeline(img, s,
                            get_landmarks=lambda: self.mesh.landmarks(img, self.detector.detect(img)),
                            get_mask=lambda: self.segmenter.get_mask(img),
                            get_faces=lambda: self.detector.detect(img),
                            bg_image=bg_image)
