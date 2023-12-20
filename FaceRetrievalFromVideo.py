from ultralytics import YOLO
import cv2 as cv
from deepface import DeepFace
from deepface.commons import distance
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import math
from numba import jit

LANDMARKS_CONFIDENCE_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = InceptionResnetV1(pretrained="vggface2", device=device).eval()
model = YOLO("yolov8n-face.pt").to(device)

# cap = cv.VideoCapture("test/biden.mp4")
cap = cv.VideoCapture(0, cv.CAP_DSHOW)


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
    embeddings = []
    labels = []

    label = str(input("Enter the full name: "))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            np.savez_compressed(
                f"database/{label}.npz",
                X=embeddings,
                y=labels,
            )
            break
        results = model.predict(
            frame,
            conf=0.5,
            imgsz=32 * 10,
            verbose=False,
            show=False,
        )[0].cpu()

        if len(results.boxes) > 0:
            for result in results:
                x, y, w, h = result.boxes.xywh.tolist()[0]
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)

                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
                # Resize face
                preprocessed = DeepFace.extract_faces(
                    aligned, target_size=(160, 160), detector_backend="skip"
                )
                with torch.no_grad():
                    embedding = (
                        resnet(
                            torch.from_numpy(preprocessed[0]["face"].copy())
                            .cuda()
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .float()
                        )
                    )[0]
                embeddings.append(embedding.cpu().numpy())
                labels.append(label)
        cv.imshow(label, frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            np.savez_compressed(
                f"database/{label}.npz",
                X=embeddings,
                y=labels,
            )
            break


if __name__ == "__main__":
    main()
