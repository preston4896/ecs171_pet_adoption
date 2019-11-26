from create_neural_network import create_network
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from Preprocessing_For_ANN import x_train, y_train, x_test, y_test
from keras.utils import to_categorical

f, axarr = plt.subplots(2, 3)

model = create_network()
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=2, class_weight='balanced')
rawProb = model.predict(x_test)

for i in range(0,5):
	lr_auc = roc_auc_score(y_test[:, i], rawProb[:, i])
	lr_fpr, lr_tpr, _ = roc_curve(y_test[:, i], rawProb[:, i])
	lr_precision, lr_recall, threshold = precision_recall_curve(y_test[:, i], rawProb[:, i])
	average_precision = sum(lr_precision) / len(lr_precision)
	axarr[0, 1].plot(lr_recall, lr_precision, marker='.', label= 'Class ' + str(i) + " (AUC:{:.3f})".format(average_precision), markersize=1)
	axarr[1, 1].plot(lr_fpr, lr_tpr, marker='.', label='Class ' + str(i) + " (AUC:{:.3f})".format(lr_auc), markersize=1)


# axis labels
axarr[0, 1].set_xlabel('Recall')
axarr[0, 1].set_ylabel('Precision')
axarr[0, 1].set_title('Precision-Recall Curve of ANN')
axarr[0, 1].legend(loc='upper right', shadow=True, fontsize='small')
axarr[1, 1].set_xlabel('False Positive Rate')
axarr[1, 1].set_ylabel('True Positive Rate')
axarr[1, 1].set_title('ROC Curve of ANN')
axarr[1, 1].legend(loc='lower right', shadow=True, fontsize='small')

from get_data_final import x_train, y_train, x_test, y_test

for i in range(0, 5):
	modelLR = LogisticRegression(solver='lbfgs', max_iter=1000)

	LRtrainingY = to_categorical(y_train)
	trainingX = x_train
	LRtrainingY = LRtrainingY[:,i]
	modelLR.fit(trainingX, LRtrainingY)
	LRtestingY = to_categorical(y_test)
	LRtestingY = LRtestingY[:,i]
	testingX = x_test

	lr_probs = modelLR.predict_proba(testingX)
	lr_probs = lr_probs[:, 1]
	lr_precision, lr_recall, threshold = precision_recall_curve(LRtestingY, lr_probs)
	lr_auc = roc_auc_score(LRtestingY, lr_probs)
	lr_fpr, lr_tpr, _ = roc_curve(LRtestingY, lr_probs)
	average_precision = sum(lr_precision) / len(lr_precision)
	axarr[0, 0].plot(lr_recall, lr_precision, marker='.', label= 'Class ' + str(i) + " (AUC:{:.3f})".format(average_precision), markersize=1)
	axarr[1, 0].plot(lr_fpr, lr_tpr, marker='.', label='Class ' + str(i) + " (AUC:{:.3f})".format(lr_auc), markersize=1)

# axis labels
axarr[0, 0].set_xlabel('Recall')
axarr[0, 0].set_ylabel('Precision')
axarr[0, 0].set_title('Precision-Recall Curve of LogisticRegression')
axarr[0, 0].legend(loc='upper right', shadow=True, fontsize='small')
axarr[1, 0].set_xlabel('False Positive Rate')
axarr[1, 0].set_ylabel('True Positive Rate')
axarr[1, 0].set_title('ROC Curve of LogisticRegression')
axarr[1, 0].legend(loc='lower right', shadow=True, fontsize='small')


import pandas as pd
import numpy as np
import lightgbm as lgb
from get_data_final import x_test, y_test
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

# Load the model
model_lgb= lgb.Booster(model_file='lgb_model.txt')

# Predict
y_prob = model_lgb.predict(x_test)
y_pred = np.argmax(y_prob, axis=1)

# Calculate the accuracy
acc = len(np.where((y_pred-y_test)==0)[0])/len(y_test)
print("Accuracy on testing set is: ", acc)


# Plot PR Curve
y_cate_test = to_categorical(y_test)
for i in range(0, 5):
	lr_auc = roc_auc_score(y_cate_test[:,i], y_prob[:, i])
	lr_fpr, lr_tpr, _ = roc_curve(y_cate_test[:,i], y_prob[:, i])
	lr_precision, lr_recall, threshold = precision_recall_curve(y_cate_test[:,i], y_prob[:, i])
	average_precision = sum(lr_precision) / len(lr_precision)
	axarr[0, 2].plot(lr_recall, lr_precision, marker='.', label= 'Class ' + str(i) + " (AUC:{:.3f})".format(average_precision), markersize=1)
	axarr[1, 2].plot(lr_fpr, lr_tpr, marker='.', label='Class ' + str(i) + " (AUC:{:.3f})".format(lr_auc), markersize=1)

# axis labels
axarr[0, 2].set_xlabel('Recall')
axarr[0, 2].set_ylabel('Precision')
axarr[0, 2].set_title('Precision-Recall Curve of LightGBM')
axarr[0, 2].legend(loc='upper right', shadow=True, fontsize='small')
axarr[1, 2].set_xlabel('False Positive Rate')
axarr[1, 2].set_ylabel('True Positive Rate')
axarr[1, 2].set_title('ROC Curve of LightGBM')
axarr[1, 2].legend(loc='lower right', shadow=True, fontsize='small')
plt.show()
#
#
