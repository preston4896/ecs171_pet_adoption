import numpy as np
from create_neural_network import create_network
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

best_params = np.load("best parameters.npy")
scores = []
for i in range(0, len(best_params)):
    model = create_network(lr=best_params[i,1],numNodes=best_params[i,2],loss=best_params[i,3])
    wrapper = KerasClassifier(build_fn=create_network, epochs=300, batch_size=32, verbose=2)
    val_scores = cross_val_score(wrapper, x_train, y_train, cv=5)
    scores.append(val_scores)
