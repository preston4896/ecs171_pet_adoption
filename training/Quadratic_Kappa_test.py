from quadratic_kappa import quadratic_kappa
from get_data_2 import *
from keras.models import load_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load the model you want to evaluate
model_name = 'model3.h5'
model = load_model(model_name)
predv = np.argmax(model.predict(x_test), axis=1)
actualy = np.argmax(y_test, axis=1)
print(quadratic_kappa(actualy, predv))
