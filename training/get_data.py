import numpy as np
from keras.utils import to_categorical

def min_max_norm(x):
    num_features = len(x[0, :])
    for i in range(0, num_features):
        min_x = min(x[:, i])
        max_x = max(x[:, i])
        x[:, i] = (x[:, i] - min_x) / (max_x - min_x)
    return x


data = np.load('new_data_shuffled.npy', allow_pickle=True)
data = np.asarray(data)

# 66% training set; 33% testing set at first, then set at 80% training set, 20% of which is cv and 20% testing set
# print(data)
train_size = int(0.8*len(data))


# Let n denote number of features
n = len(data[0]) - 1


# Normalize all the features
data[:, 0:n] = min_max_norm(data[:, 0:n])


# Training set
x_train = data[0:train_size, 0:n]
labels_train = data[0:train_size, n]
y_train = to_categorical(data[0:train_size, n], 5)

# testing set
x_test = data[train_size:, 0:n]
labels_test = data[train_size:, n]
y_test = to_categorical(data[train_size:, n], 5)

print(len(x_train[0]))
print(n)