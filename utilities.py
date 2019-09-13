import numpy as np

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

    def inverse_transform(self, X):
        return X * (self.max_array - self.min_array) + self.min_array
