"""Subprocess worker: polls a multiprocessing.Queue for BGR frames and displays them with OpenCV."""

import multiprocessing as mp

import cv2


def run(frame_queue: mp.Queue) -> None:
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        cv2.imshow("onboard_cameras", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
