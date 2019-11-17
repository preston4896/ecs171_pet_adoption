import numpy as np
from create_neural_network import create_network
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

best_params = np.load("best parameters.npy")
for i in range(0, len(best_params)):
    