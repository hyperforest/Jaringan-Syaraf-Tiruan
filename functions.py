import numpy as np
from itertools import product

def accuracy(y_true, y_pred):
	return ((y_true == y_pred).sum()) / len(y_true)

def heaviside(x, threshold=0):
	if isinstance(x, int):
		if x >= threshold:
			return 1
		return -1
	elif isinstance(x, np.ndarray):
		x[x >= threshold] = 1
		x[x < threshold] = -1
		return x

def sum_squared_error(y_true, y_pred):
	return 0.5 * np.sum(np.square(y_true - y_pred))

def cartesian_product(u, v):
	result = np.array(list(product(u, v)))
	result = np.product(result, axis=1)
	return result.reshape((u.shape[0], v.shape[0]))