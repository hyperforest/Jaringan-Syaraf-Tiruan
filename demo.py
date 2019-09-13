# import library
import pandas as pd
import numpy as np

from JST import LearningVectorQuantization as LVQ
from utilities import DataScaler, train_test_split, accuracy

# import data
df = pd.read_csv('datasets/iris.csv')
X = df.iloc[:, :4].values
y = df.iloc[:, -1].values

# set hyperparameters
n_classes = 3
input_shape = X.shape[1]
random_state = 7777777
epochs = 123
lr = 0.0001

# preprocess data
X_train, y_train, X_test, y_test = train_test_split(X, y,
    random_state=random_state)

scaler = DataScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# start training
model = LVQ(n_classes=n_classes, input_shape=input_shape)
model.train(X_train, y_train, 
    epochs=epochs, lr=lr,
    val_data=(X_test, y_test),
    random_state=random_state
    )

# overall accuracy
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
acc = accuracy(y, y_pred)
print('Accuracy using all data: {:.2f}%'.format(100 * acc))
