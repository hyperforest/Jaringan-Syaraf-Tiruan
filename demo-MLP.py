# import library
import pandas as pd
import numpy as np

from JST import MultiLayerPerceptron as MLP
from utilities import DataScaler, train_test_split
from functions import accuracy

# import data
# first 100 samples of Iris dataset
df = pd.read_csv('datasets/iris.csv')
X = df.iloc[:100, :4].values
y = df.iloc[:100, -1].values

# set hyperparameters
n_classes = 2
input_shape = X.shape[1]
random_state = 0
epochs = 100
lr = 0.1

# preprocess data
scaler = DataScaler()
X_scaled = scaler.fit_transform(X)

# start training
model = MLP(input_shape=input_shape, random_state=random_state)
model.add_layer(2, 'sigmoid')
model.add_layer(1, 'sigmoid')
model.fit(X_scaled, y, epochs=epochs, lr=lr)