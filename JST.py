import numpy as np
import os
import time

from activations import Sigmoid, Softmax, ReLU
from distance import euclidean, manhattan
from functions import heaviside, accuracy, sum_squared_error, cartesian_product
from layer import Layer
from utilities import data_shuffle


class LearningVectorQuantization:
    def __init__(self, n_classes, input_shape, distance='euclidean',
    	random_state=None):

        self.n_classes = n_classes
        self.input_shape = input_shape
        self.random_state = random_state

        if distance == 'euclidean':
        	self.distance = euclidean
        elif distance == 'manhattan':
        	self.distance = manhattan

    def get_ref_vector(self):
        return self.__rv

    def __inference(self, x, return_dist=False):
        dist = [self.distance(self.__rv[i], x) for i in range(self.n_classes)]
        dist = np.array(dist)
        
        if return_dist:
            return dist

        return np.argmin(dist)

    def predict(self, X):
        return np.array([self.__inference(X[i])
            for i in range(X.shape[0])])

    def __generate_rv_from_data(self, X, y):
        rv = np.zeros((self.n_classes, self.input_shape))
        
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            np.random.seed(self.random_state)
            idx = idx[np.random.randint(len(idx))]

            rv[i] = X[idx]	
            X = np.delete(X, idx, axis=0)
            y = np.delete(y, idx, axis=0)

        return (rv, X, y)

    def __get_accuracy(self, X, y):
        y_pred = self.predict(X)
        return accuracy(y, y_pred)

    def fit(self, X, y, epochs, lr=0.00001, val_data=None, shuffle=True):
        # adjust arguments
        self.__rv, X, y = self.__generate_rv_from_data(X, y)
        hist = {'acc':[]}
        
        if val_data != None:
            X_val, y_val = val_data
            hist['val_acc'] = []

        if shuffle:
            X, y = data_shuffle(X, y, random_state=self.random_state)

        # declare variables
        prev_acc = 0
        decay = lr / epochs
        now = time.time()

        # start training
        for e in range(1, 1 + epochs):
            for i in range(X.shape[0]):
                x, class_actual = X[i], y[i]
                class_pred = self.__inference(x)

                if class_actual != class_pred:
                    self.__rv[class_pred] -= (lr * (x - self.__rv[class_pred]))
                else:
                    self.__rv[class_pred] += (lr * (x - self.__rv[class_pred]))

            acc = self.__get_accuracy(X, y)
            hist['acc'].append(acc)

            if val_data != None:
                val_acc = self.__get_accuracy(X_val, y_val)
                hist['val_acc'].append(val_acc)
                print('Epoch [{}]\tacc: {:.4f}\tval_acc: {:.4f}'.
                    format(e, acc, val_acc))
            else:
                print('Epoch [{}]\tacc: {:.4f}'.format(e, acc))

            if e > 1:
                if acc > prev_acc:
                    lr -= decay
                elif acc < prev_acc:
                    lr += decay
            
            prev_acc = acc

        print('Elapsed time: {:.2f} s.'.format(time.time() - now))
        return hist

class MultiLayerPerceptron:
	'''
	WARNING :
	Work for binary classification only and up to 2 hidden layers only!
	'''
	def __init__(self, input_shape, random_state=None):
		self.__layer = []
		self.__input_shape = input_shape
		self.__random_state = random_state

	def add_layer(self, units, activations):
		if len(self.__layer) == 0:
			self.__layer.append(Layer(units=units, activations=activations,
				input_shape=self.__input_shape, random_state=self.__random_state))
		else:
			self.__layer.append(Layer(units=units, activations=activations,
				input_shape=self.__layer[-1].units, random_state=self.__random_state))

	def __get_accuracy(self, X, y):
		# TO DO : y true nya kan one hot lur
		y_pred = self.predict(X).ravel()
		return accuracy(y, y_pred)

	def __get_loss(self, X, y):
		return np.array([sum_squared_error(self.__feed_forward(X[i]), y[i])
            for i in range(X.shape[0])]).mean()

	def __feed_forward(self, x, return_all=False):
		result = [x]
		
		for L in self.__layer:
			x = L(x)
			result.append(x)

		if return_all:
			return result
		
		return x

	def __inference(self, x):
		x = self.__feed_forward(x)
		
		if self.__layer[-1].units == 1:
			if isinstance(self.__layer[-1].activations, Sigmoid):
				x[x >= 0.5] = 1
				x[x < 0.5] = 0
				return x
			else:
				return x
		elif isinstance(self.__layer[-1].activations, Softmax):
			return np.argmax(x)

	def predict(self, X):
		return np.array([self.__inference(X[i])
            for i in range(X.shape[0])])

	def fit(self, X, y, epochs, lr=0.01, shuffle=True):
		now = time.time()
		hist = {'loss':[], 'acc':[]}

		if shuffle:
			X, y = data_shuffle(X, y, random_state=self.__random_state)

		for e in range(1, 1 + epochs):
			for i in range(X.shape[0]):
				x, t = X[i], y[i]
				h = self.__feed_forward(x, return_all=True)
				p = h[-1]
				loss = sum_squared_error(t, p)

				# update last layer
				db2 = (p - t) * self.__layer[-1].activations(p, grad=True)
				dW2 = cartesian_product(h[-2], db2)
				self.__layer[-1].update(lr * dW2, lr * db2)

			    # update second last layer
				if len(self.__layer) == 2:
				    db1 = self.__layer[-1].activations(h[-2], grad=True)
				    db1 = db1 * np.sum(db2 * self.__layer[1].get_weights()[0], axis = 1)
				    dW1 = cartesian_product(x, db1)
				    self.__layer[-2].update(lr * dW1, lr * db1)

			loss = self.__get_loss(X, y)
			acc = self.__get_accuracy(X, y)
			hist['loss'].append(loss)
			hist['acc'].append(acc)

			print('Epoch [{}]\tloss: {:.4f}\tacc: {:.4f}'.format(e, loss, acc))

		print('Elapsed time: {:.2f} s.'.format(time.time() - now))
		return hist

	def get_weights(self):
		weights = []
		
		for L in self.__layer:
			w, b = L.get_weights()
			weights.append(w)
			weights.append(b)

		return weights

class Perceptron:
	def __init__(self, input_shape, init='random', random_state=None):
		self.__input_shape = input_shape
		self.__random_state = random_state

		if init == 'random':
			np.random.seed(random_state)
			self.w = np.random.random((input_shape, ))
			self.b = np.random.random((1, ))
		elif init == 'zero':
			self.w = np.zeros((input_shape, ))
			self.b = np.zeros((1, ))

	def __feed_forward(self, x):
		return self.w.T @ x + self.b

	def __inference(self, x):
		x = self.__feed_forward(x)
		return heaviside(x)

	def predict(self, X):
		return np.array([self.__inference(X[i])
            for i in range(X.shape[0])])

	def __get_accuracy(self, X, y):
		y_pred = self.predict(X).ravel()
		return accuracy(y, y_pred)

	def fit(self, X, y, epochs, lr=0.01, shuffle=True):
		now = time.time()
		hist = {'acc':[]}

		if shuffle:
			X, y = data_shuffle(X, y, random_state=self.__random_state)

		for e in range(1, 1 + epochs):
			for i in range(X.shape[0]):
				x, t = X[i], y[i]
				h = self.__inference(x)

				db = lr * t
				if h != t:
					self.b = self.b + db
					self.w = self.w + db * x

			acc = self.__get_accuracy(X, y)
			hist['acc'].append(acc)
			print('Epoch [{}]\tacc: {:.4f}'.
                    format(e, acc))
		
		print('Elapsed time: {:.2f} s.'.format(time.time() - now))
		return hist
