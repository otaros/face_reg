from enum import unique
from face_extract import extract_face
import os
import numpy as np
from keras_facenet import FaceNet

model = FaceNet()


def load_face(dir, required_size=(160, 160)):
    faces = []
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path, required_size)
        faces.append(face)
    return faces


def load_dataset(dir, required_size=(160, 160)):
    faces = None
    labels = None
    for subdir in os.listdir(dir):
        faces_list = []
        labels_list = []
        for filename in os.listdir(dir + subdir + "/"):
            file_path = dir + subdir + "/" + filename
            print(file_path)
            face = extract_face(file_path, required_size)
            faces_list.extend(face)
            labels_list.extend([subdir for _ in range(len(face))])
        if faces is None:
            faces = np.asarray(faces_list)
            labels = np.asarray(labels_list)
        else:
            faces = np.concatenate((faces, faces_list))
            labels = np.concatenate((labels, labels_list))
    return faces, labels


def get_embedding_face(face):
    face = face.astype("float32")
    mean, std = face.mean(), face.std()
    face = (face - mean) / std

    samples = np.expand_dims(face, axis=0)
    yhat = model.embeddings(samples)
    yhat = np.asarray(yhat)
    return yhat[0]


def calculate_average_embeddings(faces, labels):
    unique_labels = np.unique(labels)
    average_embeddings = []

    for label in unique_labels:
        label_faces = faces[labels == label]
        embeddings = []

        for face in label_faces:
            embedding = get_embedding_face(face)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)
        average_embedding = np.mean(embeddings, axis=0)
        average_embeddings.append(average_embedding)

    average_embeddings = np.array(average_embeddings)
    unique_labels = np.array(unique_labels)

    return average_embeddings, unique_labels