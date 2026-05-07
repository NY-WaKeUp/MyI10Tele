# Subprocess entry for OpenCV preview only (no mujoco import — safe with mjpython on macOS spawn).

import cv2
import numpy as np


def run(frame_queue: "object") -> None:
    win = "onboard_cameras"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.startWindowThread()
    except cv2.error:
        pass
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        out = np.ascontiguousarray(frame)
        cv2.imshow(win, out)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
