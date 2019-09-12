# import library
import pandas as pd
import numpy as np
from JST import LVQ, DataScaler, train_test_split, accuracy

# import data
df = pd.read_csv('iris.csv')
X = df.iloc[:, :4].values
y = df.iloc[:, -1].values
X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=17484)

# rescale the data
scaler = DataScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# start training
model = LVQ(n_classes=3, input_shape=4)
model.train(X_train, y_train, epochs=500,
	lr=0.00005, val_data=(X_test, y_test))

# overall accuracy
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
acc = accuracy(y, y_pred)
print('Accuracy using all data: {:.2f}%'.format(100 * acc))
