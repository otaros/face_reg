from deepface import DeepFace
import cv2 as cv

# cap = cv.VideoCapture(0, cv.CAP_DSHOW)

# while cap.isOpened():
#     ret, frame = cap.read()

result = DeepFace.extract_faces(
    "images.jpg", detector_backend="opencv", target_size=(160, 160), enforce_detection=False
)

# result = DeepFace.represent(
#     "noface.jpg",
#     model_name="Facenet",
#     detector_backend="opencv",
#     enforce_detection=False,
# )

print(len(result))
for face in result:
    if face['confidence'] < 0.9:
        continue
    cv.imshow("image", face["face"])
    cv.waitKey(0)
    cv.destroyAllWindows()
