from ultralytics import YOLO
import cv2 as cv
from deepface import DeepFace
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import faiss

resnet = InceptionResnetV1(pretrained="vggface2", device="cuda").eval()
model = YOLO("yolov8n-face.pt", task="detect")

dataset_name = "database"
X = np.load(f"database/{dataset_name}.npz", allow_pickle=True)["X"]
y = np.load(f"database/{dataset_name}.npz", allow_pickle=True)["y"]
y_unique = np.load(f"database/{dataset_name}.npz", allow_pickle=True)["target_names"]

face_embedding = faiss.IndexFlatL2(512)
face_embedding.add(X)

cap = cv.VideoCapture("test/video2.mp4")
# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
tm1 = cv.TickMeter()  # for measuring fps
tm2 = cv.TickMeter()

out = cv.VideoWriter(
    "output.avi", cv.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)
)  # for saving video


def main():
    font = cv.FONT_HERSHEY_SIMPLEX
    fps = []
    tm2.start()
    while cap.isOpened():
        if tm2.getTimeSec() >= 1:
            tm2.stop()
            tm2.reset()
            ret, frame = cap.read()
            if not ret:
                break
            tm1.start()
            result = model(
                frame, mode="predict", conf=0.45, imgsz=32 * 12, scale=0.6, verbose=False
            )
            boxes = result[0].boxes

            if len(boxes) > 0:
                for box in boxes:
                    top_left_x = int(box.xyxy.tolist()[0][0])
                    top_left_y = int(box.xyxy.tolist()[0][1])
                    bottom_right_x = int(box.xyxy.tolist()[0][2])
                    bottom_right_y = int(box.xyxy.tolist()[0][3])

                    cv.rectangle(
                        frame,
                        (top_left_x, top_left_y),
                        (bottom_right_x, bottom_right_y),
                        (0, 255, 0),
                        2,
                    )
                    face = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                    face_aligned = DeepFace.extract_faces(
                        face, target_size=(160, 160), detector_backend="skip"
                    )
                    with torch.no_grad():
                        embedding = (
                            resnet(
                                torch.from_numpy(face_aligned[0]["face"].copy())
                                .cuda()
                                .permute(2, 0, 1)
                                .unsqueeze(0)
                                .float()
                            )
                        )[0]

                    dis, index = face_embedding.search(
                        embedding.cpu().numpy().reshape(1, 512), 1
                    )

                    cv.putText(
                        frame,
                        y_unique[y[index[0][0]]] if dis[0][0] <= 0.61 else "unknown",
                        (top_left_x, top_left_y - 10),
                        font,
                        1,
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )

            tm1.stop()
            # cv.putText(
            #     frame, f"{tm1.getFPS():.2f}", (7, 35), font, 1, (100, 255, 0), 3, cv.LINE_AA
            # )
            # fps.append(tm1.getFPS())

            cv.imshow("image", frame)
            out.write(frame)
            if cv.waitKey(1) == ord("q"):
                cap.release()
                out.release()
                break
            tm2.start()
            tm1.reset()
    # print(f"max fps: {max(fps)}")
    # print(f"min fps: {min(fps)}")
    # print(f"avg fps: {sum(fps)/len(fps)}")


if __name__ == "__main__":
    main()