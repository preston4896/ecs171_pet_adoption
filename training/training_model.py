from get_data import *
import numpy as np
from keras.callbacks import LambdaCallback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from create_neural_network import create_network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Build an ANN
# The initial learning rate is quite large, when the loss starts oscillating
# I will save the model and run refine_model.py but set the learning rate to
# a smaller value and continue learning.

# Train

model = create_network()
eval_acc = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.evaluate(x_test, y_test)[1]))
model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=2, class_weight=None, callbacks=[eval_acc])

# Evaluate
wrapper = KerasClassifier(build_fn=create_network, epochs=300,batch_size=32,verbose=2)
val_scores = cross_val_score(wrapper, x_train, y_train, cv=5)
print(model.evaluate(x_test, y_test)[1])
print(np.mean(val_scores))
# Save the model
model.save('model3.h5')