import numpy as np
import os
import pickle
import time

from utilities import accuracy, data_shuffle
from distance import euclidean

class LearningVectorQuantization:
    def __init__(self, n_classes, input_shape, distance=euclidean):
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.distance = distance

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

    def __generate_rv_from_data(self, X, y, random_state=None):
        rv = np.zeros((self.n_classes, self.input_shape))
        
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            np.random.seed(random_state)
            idx = idx[np.random.randint(len(idx))]

            rv[i] = X[idx]	
            X = np.delete(X, idx, axis=0)
            y = np.delete(y, idx, axis=0)

        return (rv, X, y)

    def __get_accuracy(self, X, y):
        y_pred = self.predict(X)
        return accuracy(y, y_pred)

    def train(self, X, y, epochs, lr=0.00001, rv_from_data=True, val_data=None,
        shuffle=True, random_state=None):
        
        # adjust arguments
        if rv_from_data:
            self.__rv, X, y = self.__generate_rv_from_data(X, y,
                random_state=random_state)
        else:
            np.random.seed(random_state)
            self.__rv = np.random.random((self.n_classes, self.input_shape))

        if val_data != None:
            X_val, y_val = val_data

        if shuffle:
            X, y = data_shuffle(X, y, random_state=random_state)

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
            if val_data != None:
                val_acc = self.__get_accuracy(X_val, y_val)
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

    def save(self, filename):
    	with open(filename, 'wb') as f:
    		pickle.dump(self, f, protocol=-1)

    def load(self, filename):
    	with open(filename, 'rb') as f:
    		return pickle.load(f)