from get_data import *
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.activations import sigmoid
from keras.activations import relu
from keras.activations import softmax
from keras.losses import mean_squared_error
from keras.losses import categorical_crossentropy
from keras.losses import poisson
from keras.callbacks import LambdaCallback
import talos as ta

import os
parameters = {'lr': (1e-6,1,10),
              'num_Nodes' : [3,6,9,12],
              'Dropout' : (0,0.8,10),
              'loss_functions' : [mean_squared_error, categorical_crossentropy,poisson],
              'optimizer' : [SGD,Adam],
              'Hidden_Activation' : [sigmoid,relu],
              'Final_Activation' : [sigmoid,softmax],
                }

# Build an ANN
# The initial learning rate is quite large, when the loss starts oscillating
# I will save the model and run refine_model.py but set the learning rate to
# a smaller value and continue learning.
def pet_finder_model(x_train,y_train,x_test,y_test,params):
    model = Sequential()

    model.add(Dense(parameters['num_Nodes'], input_dim=n, activation=parameters['Hidden_Activation'], kernel_initializer='random_uniform'))
    model.add(Dense(parameters['num_Nodes'], activation=parameters['Hidden_Activation'], kernel_initializer='random_uniform'))
    model.add(Dense(5, activation=parameters['Final_Activation'], kernel_initializer='random_uniform',))
    model.compile(optimizer=parameters['optimizer'](lr=parameters['lr']), loss=parameters['loss_functions'], metrics=['accuracy'])

    # Train
    eval_acc = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.evaluate(x_test, y_test)[1]))
    out = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=2, class_weight=None, callbacks=[eval_acc])
    return out, model

ta.Scan(x=x_train, y=y_train,params=parameters,model=pet_finder_model, experiment_name='pet_finder')
# Evaluate
print(model.evaluate(x_test, y_test)[1])
# Save the model
#model.save('model2.h5')