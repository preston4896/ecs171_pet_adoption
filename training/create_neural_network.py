from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
def create_network():
    model = Sequential()
    model.add(Dense(12, input_dim=n, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dense(12, activation='sigmoid', kernel_initializer='random_uniform'))
    model.add(Dense(5, activation='sigmoid', kernel_initializer='random_uniform', ))
    model.compile(optimizer=Adam(learning_rate=0.0033), loss='categorical_crossentropy', metrics=['accuracy'])
    return model