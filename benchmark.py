"""
benchmark.py - reproduces the numbers in the report's Results section.

    python benchmark.py                                   # sample_images/, counts from BlazeFace
    python benchmark.py --images photos/ --gt gt.csv      # your own photos + true face counts

gt.csv has two columns:  file,faces   (e.g.  group1.jpg,6)

Outputs (docs/results/):
    face_counts.csv   faces found by BlazeFace multi-scale / single-pass / Haar vs ground truth
    robustness.csv    the three detectors on 12 controlled degradations of each image
    summary.csv       detection rate, extra boxes and mean time per detector
    timing.csv        mean / std run-time of every pipeline operation and style
    quality.csv       PSNR and SSIM of each filter vs the original
    detections/       annotated images (green = BlazeFace, orange = Haar) for visual checking
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd

import evaluation
import face_effects as fx
import filters
import metrics
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))


def timeit(fn, n=10):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(HERE, "sample_images"))
    ap.add_argument("--gt", default=None, help="CSV with columns file,faces")
    ap.add_argument("--out", default=os.path.join(HERE, "docs", "results"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--timing-image", default=os.path.join(HERE, "sample_images", "astronaut.png"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "detections"), exist_ok=True)
    M = P.Models()
    single = evaluation.SinglePass(M.detector)

    gt = {}
    if a.gt:
        g = pd.read_csv(a.gt)
        gt = dict(zip(g["file"], g["faces"]))
    paths = sorted(p for ext in ("png", "jpg", "jpeg") for p in glob.glob(os.path.join(a.images, f"*.{ext}")))

    # 1. Face counts vs ground truth --------------------------------------------------
    crow, images = [], []
    for p in paths:
        name = os.path.basename(p)
        img = filters.resize_max_dim(cv2.imread(p), 1400)
        mf, mt = evaluation.run_detector(M.detector, img)
        sf, st = evaluation.run_detector(single, img)
        hf, ht = evaluation.run_detector(M.haar, img)
        expected = int(gt.get(name, len(mf)))
        crow.append({"Image": name, "Size": f"{img.shape[1]}x{img.shape[0]}", "True faces": expected,
                     "BlazeFace multi-scale": len(mf), "BlazeFace single": len(sf), "Haar": len(hf),
                     "Multi ms": round(mt, 1), "Single ms": round(st, 1), "Haar ms": round(ht, 1)})
        vis = M.haar.draw_detections(M.detector.draw_detections(img, mf, label_keypoints=False), hf)
        cv2.imwrite(os.path.join(a.out, "detections", os.path.splitext(name)[0] + ".jpg"), vis)
        images.append((name, img, expected))
    counts = pd.DataFrame(crow)
    counts.to_csv(os.path.join(a.out, "face_counts.csv"), index=False)
    print(counts.to_string(index=False))
    tot = counts["True faces"].sum()
    for col in ("BlazeFace multi-scale", "BlazeFace single", "Haar"):
        found = np.minimum(counts[col], counts["True faces"]).sum()
        extra = np.maximum(counts[col] - counts["True faces"], 0).sum()
        print(f"{col:22s} found {found}/{tot} ({100 * found / max(tot, 1):.0f}%), extra boxes {extra}")

    # 2. Robustness on images that contain faces --------------------------------------
    rows = []
    for name, img, n in images:
        if n == 0:
            continue
        for r in evaluation.robustness_table(img, M.detector, M.haar, n):
            rows.append({"Image": name, **r})
    rob = pd.DataFrame(rows)
    rob.to_csv(os.path.join(a.out, "robustness.csv"), index=False)
    summ = evaluation.summarise(rows)
    pd.DataFrame([summ]).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print("\nRobustness by condition (number of images where all faces were found):")
    by = rob.groupby("Condition", sort=False)[["MediaPipe OK", "Single-pass OK", "Haar OK"]].sum()
    print(by.to_string())
    for k, v in summ.items():
        print(f"  {k}: {v}")

    # 3. Timing + quality ---------------------------------------------------------------
    img = filters.resize_max_dim(cv2.imread(a.timing_image), 1400)
    faces = M.detector.detect(img)
    lms = M.mesh.landmarks(img, faces)
    mask = M.segmenter.get_mask(img)
    ops = {
        "BlazeFace detection (single pass)": lambda: M.detector.detect(img, multiscale=False),
        "BlazeFace detection (multi-scale)": lambda: M.detector.detect(img),
        "Haar detection": lambda: M.haar.detect(img),
        "Face mesh (478 pts)": lambda: M.mesh.landmarks(img),
        "Selfie segmentation": lambda: M.segmenter.get_mask(img),
        "Brightness+contrast+saturation": lambda: filters.adjust_saturation(
            filters.adjust_contrast(filters.adjust_brightness(img, 20), 20), 20),
        "Auto-enhance (CLAHE)": lambda: filters.auto_enhance(img),
        "Unsharp mask": lambda: filters.adjust_sharpness(img, 1.0),
        "Portrait blur (compositing only)": lambda: filters.blur_background(img, mask, 15, 10),
        "Background replace (compositing only)": lambda: filters.replace_background_color(img, mask),
        "Skin smoothing": lambda: fx.beautify(img, lms, 0.6, 8),
        "AR sunglasses": lambda: fx.draw_ar_overlay(img, lms, ["Sunglasses"]),
        "Privacy pixelate": lambda: fx.anonymize_faces(img, faces, "Pixelate"),
        "Head pose + EAR": lambda: [(fx.head_pose(lm), fx.eye_state(lm)) for lm in lms],
        "Vignette": lambda: filters.apply_vignette(img, 1.5),
    }
    for s, fn in filters.STYLES.items():
        if fn is not None:
            ops[f"Style: {s}"] = (lambda f=fn: f(img))
    trows, qrows = [], []
    for op, fn in ops.items():
        mu, sd = timeit(fn, a.repeats)
        trows.append({"Operation": op, "Mean ms": round(mu, 2), "Std ms": round(sd, 2)})
        res = fn()
        if isinstance(res, np.ndarray) and res.shape == img.shape:
            qrows.append({"Operation": op, "PSNR dB": round(metrics.psnr(img, res), 2),
                          "SSIM": round(metrics.ssim(img, res), 4)})
    pd.DataFrame(trows).to_csv(os.path.join(a.out, "timing.csv"), index=False)
    pd.DataFrame(qrows).to_csv(os.path.join(a.out, "quality.csv"), index=False)
    print(f"\nTiming on {os.path.basename(a.timing_image)} ({img.shape[1]}x{img.shape[0]}):")
    print(pd.DataFrame(trows).to_string(index=False))
    print(pd.DataFrame(qrows).to_string(index=False))
    print(f"\nCSV files written to {a.out}")


if __name__ == "__main__":
    main()
