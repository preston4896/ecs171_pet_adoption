from get_data_2 import *
import numpy as np
from keras.callbacks import LambdaCallback
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from create_neural_network import create_network
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Build an ANN
# The initial learning rate is quite large, when the loss starts oscillating
# I will save the model and run refine_model.py but set the learning rate to
# a smaller value and continue learning.

# Train

model = create_network()
epochs = 1000
train_scores = []
test_scores = []
train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
test_loss = LambdaCallback(on_epoch_end=lambda batch, logs: test_scores.append(logs['val_loss']))
wrapper = KerasClassifier(build_fn=create_network, epochs=epochs,batch_size=32,verbose=0)
earlystopper = EarlyStopping(monitor='val_loss', patience=epochs/10)
model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=32, verbose=1,
          class_weight='Balanced', callbacks=[train_loss,test_loss,earlystopper])
# Evaluate
val_scores = cross_val_score(wrapper, x_train, y_train, cv=5, verbose=0, n_jobs=-1,
                             fit_params={"callbacks":[EarlyStopping(monitor='loss', patience=epochs/10)]})
model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0,
          class_weight='Balanced', callbacks=[EarlyStopping(monitor='loss', patience=epochs/10)])
print("testing accuracy:",model.evaluate(x_test, y_test)[1])
print("cv accuracy:",np.mean(val_scores))

# plot
plt.title("Learning Curve")
plt.grid()

plt.fill_between(np.linspace(0,len(train_scores),len(train_scores)), train_scores,
                  alpha=0.1, color="r")
plt.fill_between(np.linspace(0,len(test_scores),len(test_scores)), test_scores,
             alpha=0.1, color="g")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.ylim(top=max(train_scores),bottom=min(train_scores))
plt.plot(np.linspace(0,len(train_scores),len(train_scores)), train_scores, 'o-', color="r",
         label="Training score")
plt.plot(np.linspace(0,len(test_scores),len(test_scores)), test_scores, 'o-', color="g",
          label="validation score")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
legend.get_frame().set_facecolor('C0')


plt.show()


# Save the model
model.save('model3.h5')