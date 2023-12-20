from yunet import YuNet
import cv2 as cv
import numpy as np
import faiss
from deepface import DeepFace
from numba import jit
from facenet_pytorch import InceptionResnetV1
import torch
import math

LANDMARKS_CONFIDENCE_THRESHOLD = 0.5

# Instantiate YuNet
model = YuNet(
    modelPath="face_detection_yunet_2023mar.onnx",
    inputSize=[320, 320],
    confThreshold=0.86,
    nmsThreshold=0.3,
    topK=5000,
    backendId=cv.dnn.DNN_BACKEND_CUDA,
    targetId=cv.dnn.DNN_TARGET_CUDA,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = InceptionResnetV1(pretrained="vggface2", device=device).eval()

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
tm = cv.TickMeter()

dataset_name = "database"
X = np.load(f"database/{dataset_name}.npz", allow_pickle=True)["X"]
labels = np.load(f"database/{dataset_name}.npz", allow_pickle=True)["y"]
labels_unique = np.load(f"database/{dataset_name}.npz", allow_pickle=True)[
    "target_names"
]

database = faiss.IndexFlatL2(512)
database.add(X)


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
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    model.setInputSize([w, h])

    while cv.waitKey(1) < 0:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        tm.start()
        results = model.infer(frame)  # results is a tuple

        if len(results) > 0:
            bounding_boxes = []
            preprocessed = []
            # Draw results on the input image
            for det in results:
                bbox = det[0:4].astype(np.int32)
                bounding_boxes.append(bbox)
                cv.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (0, 255, 0),
                    2,
                )

                face = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

                landmarks = det[4:14].astype(np.int32).reshape((5, 2))
                right_eye = landmarks[0]
                left_eye = landmarks[1]
                aligned = alignment_procedure(face, left_eye, right_eye)

            #     preprocessed.append(
            #         DeepFace.extract_faces(
            #             aligned, target_size=(160, 160), detector_backend="skip"
            #         )[0]["face"]
            #     )

            # with torch.no_grad():
            #     embeddings = resnet(
            #         torch.tensor(preprocessed).permute(0, 3, 1, 2).float().to(device)
            #     ).cpu()
            # for i in range(len(results)):
            #     dis, idx = database.search(embeddings[i].numpy().reshape(1, 512), 1)

                # cv.putText(
                #     frame,
                #     labels_unique[labels[idx[0][0]]] if dis[0][0] <= 0.4 else "unknown",
                #     (bounding_boxes[i][0], bounding_boxes[i][1] - 10),
                #     font,
                #     1,
                #     (0, 255, 0),
                #     2,
                #     cv.LINE_AA,
                # )

        tm.stop()
        cv.putText(
            frame,
            "FPS: {:.2f}".format(tm.getFPS()),
            (0, 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
        )
        # Visualize results in a new Window
        cv.imshow("YuNet Demo", frame)

        tm.reset()


if __name__ == "__main__":
    main()
