# Face Studio Camera: demo script (about 10 minutes)

## The night before (on the demo laptop)
1. `pip install -r requirements.txt` → `python download_models.py` → `python check_setup.py` (must say **ready for the demo**).
2. Put 3–4 of **your own** photos in a folder: a selfie, a group photo, a photo with a busy background,
   and a landscape photo with a background you like.
3. Open the app once (`streamlit run app.py`) so the browser has already granted camera permission.
4. Charge the laptop and close other camera apps (Zoom, Teams, etc.).

## At the venue: 2 minutes before your slot
- Terminal 1: `streamlit run app.py` (keep the tab open on **Studio**).
- Terminal 2 ready: `python live_camera.py` (do not start it yet).
- If Wi-Fi is down, nothing changes: everything runs offline.

## Running order

| Time | Who | What to show | What to say (one line) |
|---|---|---|---|
| 0:00 | **Sparsh** | Title slide, then the About tab | "One app, 6 modules, 7-stage pipeline; all processing runs locally." |
| 0:45 | **Anmol** | Studio → Sample image → click **Portrait Pro** → drag the before/after slider | "Presets just set sliders; every stage stacks in a fixed order." |
| 1:30 | Anmol | Reset → move Brightness, Contrast, Gamma, Sharpness; open **Results & metrics** | "PSNR/SSIM and histograms quantify how much each filter changes the image." |
| 2:30 | **Shreyash** | Background → Blur, then **Custom image** (upload your background photo) | "The selfie segmenter gives a per-pixel person probability; we feather and alpha-composite." |
| 3:15 | Shreyash | AR overlays: Sunglasses + Party Hat; Beauty: skin smoothing + lip tint; Privacy: Pixelate | "Overlays are drawn in a face-local frame, so they follow size and tilt." |
| 4:15 | **Vansh** | Face Analysis → Detection on your **group photo**; open Face detection settings and untick Multi-scale | "Short-range BlazeFace misses small faces; tiling + NMS recovers them." |
| 5:00 | Vansh | Head pose · eyes · mouth; then **Haar vs MediaPipe** → Run robustness test | "Haar fails on rotated faces; BlazeFace stays robust and one pass is roughly 20–30× faster." |
| 6:00 | **Sparsh** | IPCV Lab → FFT high-pass, Noise & denoising (Median vs Gaussian on salt & pepper), Morphology | "These are the classical syllabus operations the app is built on." |
| 7:00 | Sparsh | Terminal 2: `python live_camera.py` → keys 1 (sunglasses), 5 (head pose + blink counter: blink a few times), 8 (privacy blur), 0 (portrait mode) | "Real time at the FPS shown on screen, entirely on CPU." |
| 8:30 | Sparsh | Batch tab: upload 3 photos → Process → download ZIP | "The same pipeline object powers batch, live and benchmark." |
| 9:00 | All | Conclusion slide, then questions | |

## Likely examiner questions (and short answers)
- **Why not Haar cascades?** We measured it. Haar finds frontal faces well but fails under rotation
  (30°: 1 of 7 images vs 6 of 7 for BlazeFace), gives no confidence scores or landmarks, and is
  roughly 20–30× slower than one BlazeFace pass.
- **How does portrait mode work?** Segmentation mask → Gaussian feather → `out = α·image + (1−α)·blurred`.
- **How are AR glasses placed?** Eye centres from the mesh define an origin, scale (inter-ocular distance)
  and rotation (roll); the glasses are defined in that local frame and mapped to pixels.
- **How is head pose computed?** Gram–Schmidt on 3-D landmark vectors (eye corner → eye corner,
  forehead → chin) gives a rotation matrix; yaw/pitch/roll are read from it.
- **What is EAR?** (|p2−p6| + |p3−p5|) / (2|p1−p4|) over 6 eye landmarks; below 0.20 = eye closed.
- **What is SSIM?** Luminance × contrast × structure similarity over 11×11 Gaussian windows; 1 = identical.
- **Limitations?** BlazeFace short-range misses small faces in a single pass; tiling adds some false boxes
  on very noisy images; segmentation edges blur on hair; pose pitch has a few degrees of bias.

## If something breaks
- Live tab won't start → use `python live_camera.py` (no browser needed).
- Camera busy → switch to **Upload a photo** or **Sample image**; every feature still works.
- App error about models → `python download_models.py` needs internet; that is why we run it the night before.
