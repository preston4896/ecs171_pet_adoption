import pandas as pd
import numpy as np
from keras.utils import to_categorical
np.random.seed(0)
data_train = pd.read_csv('train.csv', na_values=[""])
data_test = pd.read_csv('test.csv', na_values=[""])
data_train = data_train.to_numpy()
data_test = data_test.to_numpy()


# normalization function
def min_max_norm(x):
    num_features = len(x[0, :])
    for i in range(0, num_features):
        min_x = min(x[:, i])
        max_x = max(x[:, i])
        x[:, i] = (x[:, i] - min_x) / (max_x - min_x)
    return x


# Get rid of features that are not actual numbers
mask = np.array([1, 0, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 0, 1,
        0, 0, 1, 1], dtype=bool)
data_train = data_train[:, mask]
data_test = data_test[:, mask[0:len(mask)-1]]

# Denote n as the number of features
n = len(data_train[0])-1

# shuffle the data_train and create training set
np.random.shuffle(data_train)
train_size = int(np.floor(len(data_train)*0.66))
x_train = data_train[0:train_size, :n]
y_train = to_categorical(data_train[0:train_size, n], 5)

# Normalize testing set
x_test = data_train[train_size:, :n]
for j in range(0, n):
    minCol = min(x_train[:, j])
    maxCol = max(x_train[:, j])
    x_test[:, j] = (x_test[:, j] - minCol) / (maxCol - minCol)

# Normalize training set
x_train = min_max_norm(x_train)

# Create label testing set
y_test = to_categorical(data_train[train_size:, n], 5)
