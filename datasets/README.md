# Datasets

## `test_faces/`: face-detection test set

We used these 8 images to compare MediaPipe's face detector with the classic Haar cascade (see Section 3 of the report).

| File | True faces | Notes |
|---|---|---|
| `astronaut.png` | 1 | NASA portrait (public domain), also the app's sample image |
| `audrybt1.png` | 1 | Full-length portrait, small face |
| `mona-lisa.png` | 1 | A painting |
| `karen-and-rob.png` | 2 | Couple |
| `waynesworld2.png` | 2 | Two people, one wearing glasses |
| `bttf301.png` | 6 | Group photo |
| `addams-family.png` | 7 | Group photo, small and dark faces (the hardest image) |
| `churchill-downs.png` | 0 | No faces: checks for false alarms |
| **Total** | **20** | |

- **Source:** the 7 greyscale images are from the CMU/MIT frontal face test set as distributed in OpenCV's test data ([opencv_extra/testdata/cv/cascadeandhog/images](https://github.com/opencv/opencv_extra/tree/4.x/testdata/cv/cascadeandhog/images)). They are a standard research benchmark for face detection.
- **Ground truth:** `ground_truth.csv` lists the true number of faces per image (`file,faces`). We counted the faces by hand and checked every detected box by eye.
- **How it's used:** `benchmark.py` runs all three detectors (BlazeFace single pass, BlazeFace multi-scale, Haar) on each image and on 12 degraded versions of it (rotated, darkened, noisy, blurred, shrunk, mirrored).

```bash
python benchmark.py --images datasets/test_faces --gt datasets/test_faces/ground_truth.csv
```

## `../sample_images/`: app samples

- `astronaut.png`: the default sample photo in the app
- `coffee_no_face.png`: a photo without a face, to show how face features behave when there's nothing to find

## Using your own photos

Put your photos in any folder, write a `ground_truth.csv` with `file,faces` lines, and run:

```bash
python benchmark.py --images path/to/folder --gt path/to/folder/ground_truth.csv
```
