import numpy as np
import pandas as pd

from JST import Perceptron
from utilities import DataScaler

# import data
# first 100 samples of Iris dataset
df = pd.read_csv('datasets/iris.csv')
X = df.iloc[:100, :4].values
y = df.iloc[:100, -1].replace({0:-1}).values

scaler = DataScaler()
X_scaled = scaler.fit_transform(X)

model = Perceptron(input_shape=4, random_state=0)
model.fit(X_scaled, y, epochs=100, lr=0.001)