import numpy as np

class Linear:
	def __init__(self):
		pass

	def __call__(self, x, grad=False):
		if grad:
			return np.ones(x.shape)
		return x

class Sigmoid:
	def __init__(self):
		pass

	def __call__(self, x, grad=False):
		if grad:
			return self(x) * (1 - self(x))
		return 1./(1 + np.exp(-x))

class Softmax:
	def __init__(self):
		pass

	def __call__(self, x, grad=False):
		if grad:
			return self(x) * (1 - self(x))
		return (np.exp(x)) / (np.sum(np.exp(x)))

class ReLU:
	def __init__(self):
		pass

	def __call__(self, x, grad=False):
		if grad:
			x[x <= 0] = 0
			x[x > 0] = 1
			return x
		
		x[x < 0] = 0
		return x
