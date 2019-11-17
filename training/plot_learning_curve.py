import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback
from get_data import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from create_neural_network import create_network

plt.figure()
plt.title("Learning Curve")
wrapper = KerasClassifier(build_fn=create_network, epochs=350,batch_size=32,verbose=2)

# def CrossEntropy_Loss(ground_truth, predictions):
#     sum = 0
#     for i in range(0,len(ground_truth)):
#         for j in range(0, 5)
#             temp = temp + ground_truth[j] * np.log(predictions[j])
#         sum = sum - temp
#     return -sum
#my_loss_function = make_scorer(CrossEntropy_Loss,greater_is_better=False)
train_sizes,train_scores,test_scores = learning_curve(wrapper,x_train,y_train,cv=5,train_sizes=np.linspace(0.1,1.0,20,dtype=int))
train_scores = 1-train_scores;
test_scores = 1-test_scores;
train_scores_mean = np.mean(train_scores,axis=1)
train_scores_std = np.std(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
test_scores_std = np.std(test_scores,axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="validation score")

plt.legend(loc="best")
plt.show()