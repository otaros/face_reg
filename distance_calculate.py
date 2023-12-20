import numpy as np
from numba import jit

#Calculate Distance Function
@jit
def findCosineDistance(source_representation, test_representation):
    numerator = np.dot(source_representation, test_representation)
    denominator = np.linalg.norm(source_representation) * np.linalg.norm(
        test_representation
    )
    return 1 - (numerator / denominator)


@jit
def findEuclideanDistance(source_representation, test_representation):
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    euclidean_distance = np.linalg.norm(source_representation - test_representation)
    return euclidean_distance


@jit
def l2_normalize(x):
    return x / np.linalg.norm(x)