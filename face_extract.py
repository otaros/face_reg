import cv2 as cv
import numpy as np
from detect_fastmtcnn import FastMTCNN

detector = FastMTCNN(resize=0.5, margin=14, factor=0.5, keep_all=True, device="cuda")


def extract_face(filename, required_size=(160, 160)):
    image = cv.imread(filename)
    results = detector(image)
    faces_array = []
    # print(results)
    for x, y, w, h in results:
        x *= 2
        y *= 2
        w *= 2
        h *= 2
        face = image[y:h, x:w]
        face = cv.resize(face, required_size)
        faces_array.append(face)
    return np.asarray(faces_array)


if __name__ == "__main__":
    faces = extract_face("images.jpg")
    for face in faces:
        cv.imshow("face", face)
        cv.waitKey(0)
