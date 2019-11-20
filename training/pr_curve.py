from create_neural_network import create_network
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot


data = np.load('new_data_shuffled.npy', allow_pickle=True)
eightyPercent = int(0.8 * len(data))
trainingSet = data[0:eightyPercent, :]
testingSet = data[eightyPercent:-1, :]

trainingX = trainingSet[:, 0:-1]
rawTrainingY = trainingSet[:, -1]
trainingY = np.zeros((len(rawTrainingY), 5))
tagNums = np.array([0, 1, 2, 3, 4])
for i in range(0, len(rawTrainingY)):
	trainingY[i, :]= np.array([tagNums==rawTrainingY[i]], dtype=int)

testingX = testingSet[:, 0:-1]
rawTestingY = testingSet[:, -1]
testingY = np.zeros((len(rawTestingY), 5))
for i in range(0, len(rawTestingY)):
	testingY[i, :]= np.array([tagNums==rawTestingY[i]], dtype=int)

model = create_network()
model.fit(trainingX, trainingY, epochs=5000, batch_size=32, verbose=2, class_weight=None)
rawProb = model.predict(testingX)
for i in range(0,5):
	lr_precision, lr_recall, threshold = precision_recall_curve(testingY[:, i], rawProb[:, i])
	average_precision = sum(lr_precision) / len(lr_precision)
	pyplot.plot(lr_recall, lr_precision, marker='.', label= 'Class ' + str(i) + " (Area:{:.3f})".format(average_precision), markersize=1)

# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
pyplot.show()

for i in range(0, 5):
	modelLR = LogisticRegression(solver='lbfgs')

	LRtrainingY = np.zeros(len(rawTrainingY))
	for j in range(0, len(LRtrainingY)):
		if rawTrainingY[j] == i:
			LRtrainingY[j] = 1
		else:
			LRtrainingY[j] = 0
	modelLR.fit(trainingX, LRtrainingY)

	LRtestingY = np.zeros(len(rawTestingY))
	for j in range(0, len(LRtestingY)):
		if rawTestingY[j] == i:
			LRtestingY[j] = 1
		else:
			LRtestingY[j] = 0

	lr_probs = modelLR.predict_proba(testingX)
	lr_probs = lr_probs[:, 1]
	lr_precision, lr_recall, threshold = precision_recall_curve(LRtestingY, lr_probs)
	average_precision = sum(lr_precision) / len(lr_precision)
	pyplot.plot(lr_recall, lr_precision, marker='.', label= 'Class ' + str(i) + " (Area:{:.3f})".format(average_precision), markersize=1)

# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
pyplot.show()


