from get_data import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.losses import categorical_crossentropy
from keras.losses import poisson
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import astetik as ast
import talos as ta

import os
parameters = {'lr': [0.0005,0.001,0.0033,0.0066,0.01,0.033,0.066,0.1],
              'num_Nodes' : [6,9,12,15,18],
              'dropout' : [0.3,0.6],
              'loss_function':['mean_squared_error','categorical_crossentropy'],
              'final_activation':['sigmoid', 'softmax'],
                }

# Build an ANN
# The initial learning rate is quite large, when the loss starts oscillating
# I will save the model and run refine_model.py but set the learning rate to
# a smaller value and continue learning.
def pet_finder_model(x_train,y_train,x_test,y_test,params):
    model = Sequential()

    model.add(Dense(params['num_Nodes'], input_dim=n, activation=params['hidden_activation'], kernel_initializer='random_uniform'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['num_Nodes'], activation=params['hidden_activation'], kernel_initializer='random_uniform'))
    model.add(Dense(5, activation=params['final_activation'], kernel_initializer='random_uniform',))
    model.compile(optimizer=Adam(lr=params['lr'],decay=1e-8), loss=params['loss_function'], metrics=['accuracy'])

    # Train
    out = model.fit(x_train, y_train, epochs=400, batch_size=32, verbose=1, class_weight=None,validation_split=0.2)
    return out, model


scan_object = ta.Scan(x=x_train, y=y_train,params=parameters,model=pet_finder_model, experiment_name='pet_finder')
# Evaluate
analyze_object = ta.Analyze(scan_object)
scan_data = analyze_object.data


# heatmap correlation
analyze_object.plot_corr('val_accuracy', ['acc', 'loss', 'val_loss'])

# a four dimensional bar grid
ast.bargrid(scan_data,x='lr', y='val_accuracy',hue='num_Nodes',row='loss_function',col='dropout')
ast.bargrid(scan_data,x='lr', y='val_accuracy',hue='num_Nodes',row='final_activation',col='dropout')

#regression
analyze_object.plot_regs('loss', 'val_loss')
analyze_object.plot_regs('lr', 'loss')

best_params = analyze_object.best_params('val_accuracy', ['accuracy', 'loss', 'val_loss'])

np.save("best parameters",best_params)
np.save("scan_results",scan_data)

print(scan_data)
print(best_params)


plt.show()


# Save the model
#model.save('model2.h5')