# Models

| File | Where it comes from | Included? |
|---|---|---|
| `haarcascade_frontalface_default.xml` | OpenCV (BSD licence): the classic Viola–Jones face detector, used only as a comparison baseline | Yes |
| `blaze_face_short_range.tflite` | Google MediaPipe face detector | Downloaded by `python download_models.py` |
| `face_landmarker.task` | Google MediaPipe 478-point face mesh | Downloaded by `python download_models.py` |
| `selfie_segmenter.tflite` | Google MediaPipe person / background segmentation | Downloaded by `python download_models.py` |

The three MediaPipe files (about 5 MB in total) come from Google's official model storage and are not stored in this repository. Run `python download_models.py` once while you're online; after that, the app works offline.
