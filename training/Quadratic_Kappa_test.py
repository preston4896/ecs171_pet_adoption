from quadratic_kappa import *
from get_data import *
from keras.models import load_model
from keras import backend as kr
from data_prep import *
from keras.callbacks import LambdaCallback
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load the model you want to evaluate
model_name = 'model0.h5'
model = load_model(model_name)
predv = np.argmax(model.predict(x_test), axis=1)
print(quadratic_weighted_kappa(actuals_test, predv))
