from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from get_data import *
def create_network(lr=0.0033,numNodes=12,loss='mean_squared_error'):
    model = Sequential()
    model.add(Dense(numNodes, input_dim=n, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dense(numNodes, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dense(5, activation='sigmoid', kernel_initializer='random_uniform'))
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=['accuracy'])
    return model