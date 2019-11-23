from Preprocessing_For_ANN import *
import matplotlib.pyplot as plt
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from talos.model.early_stopper import early_stopper
from keras.callbacks import LambdaCallback
from create_neural_network import create_network

epochs = 1000

# Build an ANN
# The initial learning rate is quite large, when the loss starts oscillating
# I will save the model and run refine_model.py but set the learning rate to
# a smaller value and continue learning.


wrapper = KerasClassifier(build_fn=create_network, epochs=epochs)
temp = cross_val_score(wrapper, x_train, y_train, cv=5, verbose=1, fit_params={'class_weight' : ['Balanced']})
cv_acc = np.mean(temp)
final_model = create_network()
train_scores = []
val_scores = []
train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
val_loss = LambdaCallback(on_epoch_end=lambda batch, logs: val_scores.append(logs['val_loss']))
earlystopper = EarlyStopping(monitor='val_loss', patience=epochs/10)
final_model.fit(x_train,y_train,epochs=epochs, validation_split=0.2, batch_size=32, verbose=1, class_weight='Balanced',
                callbacks=[train_loss, val_loss])
#retrain for all training data
test_scores = []
final_model = create_network()
test_loss = LambdaCallback(on_epoch_end=lambda batch, logs: test_scores.append(final_model.evaluate(x_test, y_test)[0]))
final_model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0,class_weight='Balanced',callbacks=[test_loss])
print("testing accuracy:",final_model.evaluate(x_test, y_test)[1])
print("cross validation accuracy:", cv_acc)

plt.figure()
plt.title("Learning Curve")
plt.grid()

plt.fill_between(np.linspace(0,len(train_scores),len(train_scores)), train_scores,
                  alpha=0.1, color="r")
plt.fill_between(np.linspace(0,len(val_scores),len(val_scores)), val_scores,
             alpha=0.1, color="g")
plt.fill_between(np.linspace(0,len(test_scores),len(test_scores)), test_scores,
             alpha=0.1, color="b")

plt.ylabel("loss")
plt.xlabel("epochs")
plt.ylim(top=max(train_scores),bottom=min(train_scores))
plt.plot(np.linspace(0,len(train_scores),len(train_scores)), train_scores, 'o-', color="r",
         label="Training score")
plt.plot(np.linspace(0,len(val_scores),len(val_scores)), val_scores, 'o-', color="g",
          label="validation score")
plt.plot(np.linspace(0,len(test_scores),len(test_scores)), test_scores, 'o-', color="b",
          label="test score")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
legend.get_frame().set_facecolor('C0')

plt.show()