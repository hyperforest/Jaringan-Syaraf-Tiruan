import numpy as np

def manhattan(u, v):
	return np.sum(np.abs(u - v))

def euclidean(u, v):
	return np.sqrt(np.sum(np.square(u - v)))

def accuracy(y_true, y_pred):
	return ((y_true == y_pred).sum()) / len(y_true)

def train_test_split(X, y, test_size=0.2, random_state=None):
	n_classes = len(np.unique(y))
	index_class = [np.where(y == i) for i in range(n_classes)]

	X_train, y_train, X_test, y_test = [], [], [], []
	for i in range(n_classes):
		n_samples = len(index_class[i][0])
		test_sample = int(test_size * n_samples)

		index_range = np.arange(n_samples)
		np.random.seed(random_state)
		np.random.shuffle(index_range)
		index_test = index_range[:test_sample]
		index_train = index_range[test_sample:]

		X_train.append(X[index_class[i]][index_train])
		y_train.append(y[index_class[i]][index_train])
		X_test.append(X[index_class[i]][index_test])
		y_test.append(y[index_class[i]][index_test])

	X_train, y_train = np.vstack(X_train), np.hstack(y_train)
	X_train, y_train = data_shuffle(X_train, y_train, random_state=random_state)
	
	X_test, y_test = np.vstack(X_test), np.hstack(y_test)
	X_test, y_test = data_shuffle(X_test, y_test, random_state=random_state)

	return (X_train, y_train, X_test, y_test)

def data_shuffle(X, y, random_state=None):
	index_shuffle = np.arange(X.shape[0])
	np.random.seed(random_state)
	np.random.shuffle(index_shuffle)
	return (X[index_shuffle], y[index_shuffle])

class DataScaler():
	def __init__(self):
		pass

	def fit(self, X):
		self.min_array = np.min(X, axis=0)
		self.max_array = np.max(X, axis=0)

	def transform(self, X):
		return (X - self.min_array) / (self.max_array - self.min_array)

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)

class LVQ():
	def __init__(self, n_classes, input_shape):
		self.n_classes = n_classes
		self.input_shape = input_shape

	def get_ref_vector(self):
		return self.__rv

	def __inference(self, x, distance=euclidean, return_dist=False):
		dist = [distance(self.__rv[i], x) for i in range(self.n_classes)]
		dist = np.array(dist)
		
		if return_dist:
			return dist

		return np.argmin(dist)

	def predict(self, X):
		return np.array([self.__inference(X[i]) for i in range(X.shape[0])])

	def __generate_rv_from_data(self, X, y):
		rv = np.zeros((self.n_classes, self.input_shape))
		
		for i in range(self.n_classes):
			idx = np.where(y == i)[0][0]
			rv[i] = X[idx]
			X = np.delete(X, idx, axis = 0)
			y = np.delete(y, idx, axis = 0)

		return (rv, X, y)

	def __get_accuracy(self, X, y):
		y_pred = self.predict(X)
		return accuracy(y, y_pred)

	def train(self, X, y, epochs, lr=0.1, distance=euclidean,
		rv_from_data=True, val_data=None,
		shuffle=True, random_state=None):
		
		if rv_from_data:
			self.__rv, X, y = self.__generate_rv_from_data(X, y)
		else:
			np.random.seed(random_state)
			self.__rv = np.random.random((self.n_classes, self.input_shape))

		if val_data != None:
			X_val, y_val = val_data

		if shuffle:
			X, y = data_shuffle(X, y, random_state=random_state)

		for e in range(epochs):
			for i in range(X.shape[0]):
				x, class_actual = X[i], y[i]
				class_pred = self.__inference(x, distance=distance)

				if class_actual != class_pred:
					self.__rv[class_pred] -= (lr * (x - self.__rv[class_pred]))
				else:
					self.__rv[class_pred] += (lr * (x - self.__rv[class_pred]))

			acc = self.__get_accuracy(X, y)
			if val_data != None:
				val_acc = self.__get_accuracy(X_val, y_val)
				print('Epoch [{}]\tacc: {:.4f}\tval_acc: {:.4f}'.format(e+1, acc, val_acc))
			else:
				print('Epoch [{}]\tacc: {:.4f}'.format(e+1, acc))
