from get_data_2 import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import regularizers
from keras.losses import mean_squared_error
from keras.losses import categorical_crossentropy
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import astetik as ast
import talos as ta
from talos.model.early_stopper import early_stopper

import os
parameters = {'lr': [0.005,0.01,0.03,0.06,0.1,0.13],
              'num_Nodes' : [9,12,15,18,21],
              'dropout' : [1,0.1,0.2,0.3,0.4,0.5],
              'regularizer':[None,regularizers.l2(0.01)],
              'loss_function':['mean_squared_error','categorical_crossentropy','poisson'],
              'final_activation':['sigmoid','softmax']
                }

# Build an ANN
# The initial learning rate is quite large, when the loss starts oscillating
# I will save the model and run refine_model.py but set the learning rate to
# a smaller value and continue learning.
def pet_finder_model(x_train,y_train,x_test,y_test,params):
    model = Sequential()

    model.add(Dense(params['num_Nodes'], input_dim=n, activation='sigmoid', kernel_regularizer=params['regularizer'], kernel_initializer='random_uniform'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['num_Nodes'], activation='sigmoid', kernel_regularizer=params['regularizer'], kernel_initializer='random_uniform'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(5, activation=params['final_activation'], kernel_regularizer=params['regularizer'], kernel_initializer='random_uniform',))


    model.compile(optimizer=Adam(lr=params['lr'],decay=1e-8), loss=params['loss_function'], metrics=['accuracy'])

    # Train
    out = model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=1, class_weight=None,validation_data=[x_test,y_test],
                    callbacks=[early_stopper(mode=[0,100]),early_stopper(monitor='val_accuracy',mode=[0,200])])

    return out, model


scan_object = ta.Scan(x=x_train, y=y_train,x_val=x_test,y_val=y_test, params=parameters,model=pet_finder_model, experiment_name='pet_finder')
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