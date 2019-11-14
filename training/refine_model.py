from keras.models import load_model
from keras import backend as kr
from get_data import *
from keras.callbacks import LambdaCallback
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Change the name of the model you want to keep training
model_name = 'model2.h5'
# If the loss is decreasing slowly, try a larger learning rate
# If the loss is oscillating each epoch, it's probably okay since we are doing stochastic
# gradient descent. As long as it is decreasing in a long term, it's fine
# If the loss is oscillating abnormally, try a smaller learning rate
# Also try different batch_size if you want to
l_r = 0.001

# Load the model
model = load_model(model_name)

# Set a new learning rate
kr.set_value(model.optimizer.lr, l_r)

# Continue Training
# I define the callback function of loss because I want to see more decimal places
# Add eval_acc to callbacks enable you to see the accuracy on testing set every epochs
eval_loss = LambdaCallback(on_epoch_end=lambda batch, logs: print("The loss on training set: ", model.evaluate(x_train, y_train)[0]))
eval_acc = LambdaCallback(on_epoch_end=lambda batch, logs: print("The accuracy on testing set: ", model.evaluate(x_test, y_test)[1]))
model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=2, callbacks=[eval_loss, eval_acc])

# Evaluate
print(model.evaluate(x_test, y_test)[1])

# Save the model
model.save(model_name)