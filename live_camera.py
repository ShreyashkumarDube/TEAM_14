"""
live_camera.py
----------------------------------------------------------------------
Stand-alone REAL-TIME webcam window (no browser needed) - the most
reliable option for a live demo.

    python live_camera.py            # default webcam (index 0)
    python live_camera.py --cam 1    # another camera

Keys
    1-9, 0, a-e   switch effect (list printed in the terminal & on screen)
    n / p         next / previous effect
    m             toggle mirror
    s             save snapshot to saved_images/
    h             show / hide help
    q or Esc      quit

Needs the GUI build of OpenCV (opencv-python, not -headless).
----------------------------------------------------------------------
"""

import argparse
import os
import sys
from datetime import datetime

import cv2

from live_effects import LIVE_MODES, LiveProcessor
from pipeline import Models

KEYS = "1234567890abcde"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.cam}. Try --cam 1, or close other apps using the webcam.")

    proc = LiveProcessor(Models())
    idx, show_help = 0, True
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_images")
    os.makedirs(save_dir, exist_ok=True)

    print("\nFace Studio Camera - live window")
    for k, m in zip(KEYS, LIVE_MODES):
        print(f"  [{k}] {m}")
    print("  [n/p] next/prev  [m] mirror  [s] snapshot  [h] help  [q] quit\n")

    win = "Face Studio Camera - Live"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error:
        sys.exit("This OpenCV build has no GUI. Run:  pip uninstall -y opencv-python-headless && "
                 "pip install opencv-python")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not received - exiting.")
            break
        proc.mode = LIVE_MODES[idx]
        try:
            out = proc.process(frame)
        except Exception as exc:  # noqa: BLE001  (keep the demo alive)
            out = frame
            cv2.putText(out, f"Error: {exc}"[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if show_help:
            for i, (k, m) in enumerate(zip(KEYS, LIVE_MODES)):
                col = (0, 255, 200) if i == idx else (230, 230, 230)
                cv2.putText(out, f"[{k}] {m}", (out.shape[1] - 330, 24 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, f"[{k}] {m}", (out.shape[1] - 330, 24 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        ch = chr(key) if key < 128 else ""
        if ch and ch in KEYS[:len(LIVE_MODES)]:
            idx = KEYS.index(ch)
        elif ch == "n":
            idx = (idx + 1) % len(LIVE_MODES)
        elif ch == "p":
            idx = (idx - 1) % len(LIVE_MODES)
        elif ch == "m":
            proc.mirror = not proc.mirror
        elif ch == "h":
            show_help = not show_help
        elif ch == "s":
            path = os.path.join(save_dir, f"live_{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(path, out)
            print("Saved", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
