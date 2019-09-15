import numpy as np
from activations import Sigmoid, Softmax, ReLU

class Layer:
	def __init__(self, units, activations, input_shape=None, random_state=None):
		self.__random_state = random_state

		np.random.seed(random_state)
		self.__W = np.random.random((input_shape, units))
		self.__b = np.random.random((units, ))
		self.units = units

		if activations == 'sigmoid':
			self.activations = Sigmoid()
		elif activations == 'softmax':
			self.activations = Softmax()
		elif activations == 'relu':
			self.activations = ReLU()

	def __call__(self, x, activated=True):
		x = self.__W.T @ x + self.__b
		if activated:
			return self.activations(x)
		return x

	def update(self, dw, db):
		self.__W -= dw
		self.__b -= db

	def get_weights(self):
		return (self.__W, self.__b)