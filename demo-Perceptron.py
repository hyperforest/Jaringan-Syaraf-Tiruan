import numpy as np
from JST import Perceptron

# data from Fausett
X = [[1, 1, 1, 1], [-1, 1, -1, -1], [1, 1, 1, -1], [1, -1, -1, 1]]
X = np.array(X)
y = np.array([1, 1, -1, -1])

model = Perceptron(input_shape=4, init='zero')
model.fit(X, y, epochs=5, lr=1.)