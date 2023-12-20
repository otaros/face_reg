from threading import Thread
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2 as cv
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
cap = cv.VideoCapture(0, cv.CAP_DSHOW)


class FastMTCNN(object):
    """Fast MTCNN implementation."""

    def __init__(self, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.

        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frame):
        """Detect faces in a single frame using MTCNN."""
        frame = cv.resize(
            frame,
            (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)),
        )

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn.detect(frame)

        faces_locations = []
        if boxes is not None:
            for i in range(len(boxes)):
                if probs[i] < 0.92:
                    continue
                box = [int(b) for b in boxes[i]]
                faces_locations.append(box)

        return faces_locations


def main():
    detector = FastMTCNN(
        resize=0.125,
        margin=14,
        factor=0.709,
        keep_all=True,
        device=device,
        post_process=False,
        selection_method="largest",
    )
    font = cv.FONT_HERSHEY_SIMPLEX

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    max_fps = 0
    min_fps = 1000
    avg_fps = []

    while cap.isOpened():
        ret, frame = cap.read()
        print(frame.shape)
        new_frame_time = time.time()
        locations = detector(frame)
        for x, y, w, h in locations:
            cv.rectangle(frame, (x * 8, y * 8), (w * 8, h * 8), (0, 255, 0), 2)
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # converting the fps into integer
        avg_fps.append(fps)

        fps = int(fps)

        if fps > max_fps:
            max_fps = fps
        if fps < min_fps and fps > 0:
            min_fps = fps

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
        cv.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)
        cv.imshow("frame", frame)
        if (cv.waitKey(1) & 0xFF) == ord("q"):
            print("Max FPS: ", max_fps)
            print("Min FPS: ", min_fps)
            print("Avg FPS: ", sum(avg_fps) / len(avg_fps))
            break


if __name__ == "__main__":
    main()
