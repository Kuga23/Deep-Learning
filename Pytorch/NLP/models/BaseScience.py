import numpy as np


def cosine(vector1, vector2):
	return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2)) 

def euclidean(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)