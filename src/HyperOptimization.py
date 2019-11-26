from Preprocessing_For_ANN import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import astetik as ast
import talos as ta
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from talos.model.early_stopper import early_stopper
from keras.callbacks import LambdaCallback
from create_neural_network import create_network

# This script runs the grid search and plot the necessary graphs. It saves the best parameter output by the grid search in a file.
# If you think it is a better set of hyperparameters than the ones in create_neural_network, put those as the default parameters
# of create_network, which is used by test_ANN_model. This script does not automatically do that because users have to analyze
# the learning curve and then make a judgement.



# This is the set of search space for grid search
parameters = {'lr': [0.001,0.0033,0.0066,0.01,0.015],
              'num_Nodes': [9,12,15,18,21],
              'dropout': [0.1,0.2,0.3,0.4,0.5],
              'loss_function': ['categorical_crossentropy','mean_squared_error','poisson'],
              'final_activation': ['sigmoid']
              }
epochs = 1000


# model builder for Talos scan function
def pet_finder_model(x_train, y_train, x_test, y_test, params):
    model = Sequential()

    model.add(Dense(params['num_Nodes'], input_dim=n, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['num_Nodes'], activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(5, activation=params['final_activation'], kernel_initializer='random_uniform', ))

    model.compile(optimizer=Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss=params['loss_function'],
                  metrics=['accuracy'])

    # Train
    out = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=32, verbose=1, class_weight='Balanced',
                    validation_split=0.2,
                    callbacks=[early_stopper(epochs=1000, monitor='val_loss', mode='moderate'),
                               early_stopper(epochs=1000, monitor='val_accuracy', mode='moderate')])

    return out, model

# Running Talos scan. plot correlation heatmap and bar plot. Return the list of parameters in descending order based on validation accuracy
def Optimization():
    scan_object = ta.Scan(x=x_train, y=y_train, params=parameters, model=pet_finder_model, val_split=0,
                          experiment_name='pet_finder')
    # Evaluate
    analyze_object = ta.Analyze(scan_object)
    scan_data = analyze_object.data

    # heatmap correlation
    analyze_object.plot_corr('val_accuracy', ['accuracy', 'loss', 'val_loss'])

    # a four dimensional bar grid
    ast.bargrid(scan_data, x='lr', y='val_accuracy', hue='num_Nodes', row='loss_function', col='dropout')
    list_of_parameters = analyze_object.table('val_loss', ['accuracy', 'loss', 'val_loss'], 'val_accuracy')
    return list_of_parameters


best_parameters = Optimization()
best_parameters = best_parameters.iloc[[0]]

# Build the new model based on the best parameters
def build_fn(lr=best_parameters['lr'].iloc[0], num_Nodes=best_parameters['num_Nodes'].iloc[0],
             dropout=best_parameters['dropout'].iloc[0], final_activation=best_parameters['final_activation'].iloc[0],
             loss_function=best_parameters['loss_function'].iloc[0]):
    model = Sequential()
    model.add(Dense(num_Nodes, input_dim=n, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(num_Nodes, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(5, activation=final_activation, kernel_initializer='random_uniform'))
    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss=loss_function,
                  metrics=['accuracy'])
    return model


wrapper = KerasClassifier(build_fn=build_fn, epochs=epochs)
temp = cross_val_score(wrapper, x_train, y_train, cv=5, verbose=1, fit_params={'class_weight': ['Balanced']})
cv_acc = np.mean(temp)
final_model = build_fn()
train_scores = []
val_scores = []
train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
val_loss = LambdaCallback(on_epoch_end=lambda batch, logs: val_scores.append(logs['val_loss']))
earlystopper = EarlyStopping(monitor='val_loss', patience=epochs / 10)
final_model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, batch_size=32, verbose=1,
                class_weight='Balanced',
                callbacks=[train_loss, val_loss])

# retrain for all training data
test_scores = []
final_model = build_fn()
test_loss = LambdaCallback(on_epoch_end=lambda batch, logs: test_scores.append(final_model.evaluate(x_test, y_test)[0]))
final_model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0, class_weight='Balanced',
                callbacks=[test_loss])
print("testing accuracy:", final_model.evaluate(x_test, y_test)[1])
print("cross validation accuracy:", cv_acc)
print("best parameters:", best_parameters)
pd.to_pickle(best_parameters,'best_parameters_output_by_HyperOptimization') # Saves the best hyperparameters as a pandas dataframe file.
# Plot Learning Curve
plt.figure()
plt.title("Learning Curve")
plt.grid()

plt.fill_between(np.linspace(0, len(train_scores), len(train_scores)), train_scores,
                 alpha=0.1, color="r")
plt.fill_between(np.linspace(0, len(val_scores), len(val_scores)), val_scores,
                 alpha=0.1, color="g")
plt.fill_between(np.linspace(0, len(test_scores), len(test_scores)), test_scores,
                 alpha=0.1, color="b")

plt.ylabel("loss")
plt.xlabel("epochs")
plt.ylim(top=max(train_scores), bottom=min(train_scores))
plt.plot(np.linspace(0, len(train_scores), len(train_scores)), train_scores, 'o-', color="r",
         label="Training score")
plt.plot(np.linspace(0, len(val_scores), len(val_scores)), val_scores, 'o-', color="g",
         label="validation score")
plt.plot(np.linspace(0, len(test_scores), len(test_scores)), test_scores, 'o-', color="b",
         label="test score")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
legend.get_frame().set_facecolor('C0')

plt.show()
