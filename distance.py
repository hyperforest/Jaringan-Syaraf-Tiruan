import numpy as np

def manhattan(u, v):
    if len(u.shape) == 1:
        return np.sum(np.abs(u - v))
    elif len(u.shape) == 2:
        return np.sum(np.abs(u - v), axis=1)

def euclidean(u, v):
    if len(u.shape) == 1:
        return np.sqrt(np.sum(np.square(u - v)))
    elif len(u.shape) == 2:
        return np.sqrt(np.sum(np.square(u - v), axis=1))
