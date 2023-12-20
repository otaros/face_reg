from distance_calculate import *
from numba import jit


# Find Function
@jit(parallel=True)
def find(face_embedding, database, distance_metric="eulidean", verbose=True):
    min = 1000
    index = -1
    verified = False
    if distance_metric == "cosine":
        for i in range(len(database)):
            dist = findCosineDistance(face_embedding, database[i])
            if dist < min:
                min = dist
                index = i
        if min <= 0.4:
            verified = True
    elif distance_metric == "euclidean":
        for i in range(len(database)):
            dist = findEuclideanDistance(face_embedding, database[i])
            if dist < min:
                min = dist
                index = i
        if min <= 16:
            verified = True
    elif distance_metric == "euclidean_l2":
        for i in range(len(database)):
            dist = findEuclideanDistance(
                l2_normalize(database[i]), l2_normalize(face_embedding)
            )
            if dist < min:
                min = dist
                index = i
        if min <= 0.75:
            verified = True
    if verbose:
        print("Min distance: ", min)
    return index, verified
