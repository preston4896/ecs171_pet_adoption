from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from get_data_2 import *



# Hpyerparameter Optimization:
# round 1: lr=0.0033,num_Nodes=12,dropout=1,loss='mean_Squared_error'
# round 2: lr=0.0066,num_Nodes=18, dropout=0.3,final_activation='sigmoid', loss='categorical_crossentropy'
# round 3:
def create_network(lr=0.0066, num_Nodes=18, dropout=0.3, final_activation='sigmoid',
                   loss_function='categorical_crossentropy', n=n):

    model = Sequential()
    model.add(Dense(num_Nodes, input_dim=n, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(num_Nodes, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(5, activation=final_activation, kernel_initializer='random_uniform'))
    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss=loss_function,
                  metrics=['accuracy'])
    return model
