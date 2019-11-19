import numpy as np
import pandas as pd
from create_neural_network import create_network
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from get_data import *
best_params = np.load("round 1 best parameters.npy",allow_pickle=True)
best_params = pd.DataFrame(best_params)
best_params.columns = ['loss','final_activation','dropout','num_Nodes','epochs','lr','index']

model = create_network(lr=best_params['lr'][0],dropout=best_params['dropout'][0],
    final_activation=best_params['final_activation'][0],num_Nodes=best_params['num_Nodes'][0],loss=best_params['loss'][0])
wrapper = KerasClassifier(build_fn=create_network, epochs=400, batch_size=32, verbose=2)
val_scores = cross_val_score(wrapper, x_train, y_train, cv=5)
print("cv score:",np.mean(val_scores))
model.fit(x_train, y_train, epochs=400, batch_size=32, verbose=2, class_weight='balanced')
print("testing set score:",model.evaluate(x_test, y_test)[1])
