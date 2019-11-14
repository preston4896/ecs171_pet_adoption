from get_data import *
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import LambdaCallback
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Build an ANN
# The initial learning rate is quite large, when the loss starts oscillating
# I will save the model and run refine_model.py but set the learning rate to
# a smaller value and continue learning.
model = Sequential()
model.add(Dense(10, input_dim=n, activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Dense(10, activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Dense(5, activation='sigmoid', kernel_initializer='random_uniform',))
model.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error', metrics=['accuracy'])

# Train
eval_acc = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.evaluate(x_test, y_test)[1]))
model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=2, class_weight=None, callbacks=[eval_acc])

# Evaluate
print(model.evaluate(x_test, y_test)[1])

# Save the model
model.save('model2.h5')