import math
import numpy as np
import faiss
import cv2 as cv
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from deepface import DeepFace
from numba import jit
import time

LANDMARKS_CONFIDENCE_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = InceptionResnetV1(pretrained="vggface2", device=device).eval()
model = YOLO("yolov8n-face.pt")

dataset_name = "database"
X = np.load(f"database/{dataset_name}.npz", allow_pickle=True)["X"]
labels = np.load(f"database/{dataset_name}.npz", allow_pickle=True)["y"]
labels_unique = np.load(f"database/{dataset_name}.npz", allow_pickle=True)[
    "target_names"
]

database = faiss.IndexFlatL2(512)
database.add(X)

# cap = cv.VideoCapture("test/video2.mp4")
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
tm = cv.TickMeter()  # for measuring fps

# out = cv.VideoWriter(
#     "output.avi", cv.VideoWriter_fourcc(*"MJPG"), 20, (640, 480)
# )  # for saving video


@jit
def findEuclideanDistance(source_representation, test_representation):
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    euclidean_distance = np.linalg.norm(source_representation - test_representation)
    return euclidean_distance


@jit
def alignment_procedure(img, left_eye, right_eye):
    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = findEuclideanDistance(left_eye, point_3rd)
    b = findEuclideanDistance(right_eye, point_3rd)
    c = findEuclideanDistance(right_eye, left_eye)

    # -----------------------

    # apply cosine rule

    if (
        b != 0 and c != 0
    ):  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

    M = cv.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
    img = cv.warpAffine(img, M, img.shape[1::-1])

    # -----------------------

    return img  # return img anyway


def main():
    font = cv.FONT_HERSHEY_SIMPLEX
    fps = []

    while cap.isOpened():
        tm.start()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=0.5,
            imgsz=32 * 10,
            verbose=False,
            show=False,
        )[0].cpu()

        if len(results.boxes) > 0:
            bounding_boxes = []
            preprocessed = []
            for result in results:
                x, y, w, h = result.boxes.xywh.tolist()[0]
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                bounding_boxes.append([x, y, w, h])

                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Crop the face
                face = frame[y : y + h, x : x + w]

                # Align the face
                # Tuple of x,y and confidence for left eye
                left_eye = result.keypoints.xy[0][0], result.keypoints.conf[0][0]
                # Tuple of x,y and confidence for right eye
                right_eye = result.keypoints.xy[0][1], result.keypoints.conf[0][1]
                if (
                    left_eye[1] > LANDMARKS_CONFIDENCE_THRESHOLD
                    and right_eye[1] > LANDMARKS_CONFIDENCE_THRESHOLD
                ):
                    aligned = alignment_procedure(
                        face, left_eye[0].numpy(), right_eye[0].numpy()
                    )
                else:
                    aligned = face

                # resize the face
                preprocessed.append(
                    DeepFace.extract_faces(
                        aligned, target_size=(160, 160), detector_backend="skip"
                    )[0]["face"]
                )

            with torch.no_grad():
                embeddings = resnet(
                    torch.tensor(preprocessed).permute(0, 3, 1, 2).float().to(device)
                ).cpu()

            for i in range(len(results.boxes)):
                dis, idx = database.search(embeddings[i].numpy().reshape(1, 512), 1)

                cv.putText(
                    frame,
                    labels_unique[labels[idx[0][0]]] if dis[0][0] <= 0.36 else "unknown",
                    (bounding_boxes[i][0], bounding_boxes[i][1] - 10),
                    font,
                    1,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

        tm.stop()
        # cv.putText(
        #     frame, f"{tm.getFPS():.2f}", (7, 35), font, 1, (100, 255, 0), 3, cv.LINE_AA
        # )
        # fps.append(tm.getFPS())

        cv.imshow("image", frame)
        # out.write(frame)
        if cv.waitKey(1) == ord("q"):
            cap.release()
            # out.release()
            break
        tm.reset()
    # print(f"max fps: {max(fps)}")
    # print(f"min fps: {min(fps)}")
    # print(f"avg fps: {sum(fps)/len(fps)}")


if __name__ == "__main__":
    main()
