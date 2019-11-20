import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn import metrics

# This file need sentiment fold and train.csv to run


# Read Data
data = pd.read_csv('train.csv')


# Show some data
print("Features in the CSV File: ")
print(data.columns)

print("\nLabels Types: ")
print(pd.unique(data.AdoptionSpeed))

print("\nNumbers of samples in each category: ")
print(data.groupby('AdoptionSpeed')['Type'].count())

print(len(data))


# Add Sentiment Data
petID = pd.read_csv('train.csv', sep=',', usecols=['PetID']).values
data_temp = np.zeros((len(data), 2), dtype=float)
fileNotFound = 0
fileMissingIndex = []
magnitude = 0
score = 0
for i in range(0, len(data)):
	try:
		fileName = 'train_sentiment/' + petID[i][0] + '.json'
		with open(fileName) as json_file:
		    info = json.load(json_file)
		    data_temp[i, -2] = info['documentSentiment']['magnitude']
		    data_temp[i, -1] = info['documentSentiment']['score']
		    magnitude+=info['documentSentiment']['magnitude']
		    score+=info['documentSentiment']['score']
	except:
		fileMissingIndex.append(i)
		fileNotFound += 1

		# uncomment to see which files are missing
		#print("Could not find {}.json".format(petID[i][0]))
		
averageMagnitude = magnitude / (len(data) - fileNotFound)
averageScore = score / (len(data) - fileNotFound)

data_temp[fileMissingIndex, -2] = averageMagnitude
data_temp[fileMissingIndex, -1] = averageScore

data['magnitude']=data_temp[:, -2]
data['score']=data_temp[:, -1]
print(data.head())

# Drop none numerical features
data = data.drop(['Name', 'State', 'RescuerID', 'Description', 'PetID'], axis=1)

# Shuffle the data
data = data.sample(frac=1)
print(data.head())

# Let vs denote Validation Size: 10% validation set, 10% testing set, 80% training set
vs = 0.1
vs = int(len(data)*vs)
train = data[:-2*vs]
valid = data[-2*vs:]
test = data[-vs:]

# LGB model training:
# Drop labels
feats = data.columns.drop('AdoptionSpeed')
train = lgb.Dataset(train[feats], label=train['AdoptionSpeed'])
val = lgb.Dataset(valid[feats], label=valid['AdoptionSpeed'])

param = {'num_leaves': 128, 'objective': 'multiclass', 'num_class':5, 'metric': 'multi_logloss'}
num_rounds = 2500

model_lgb = lgb.train(param, train, num_rounds, valid_sets=[val], early_stopping_rounds=10, verbose_eval=False)

ypred = np.argmax(model_lgb.predict(test[feats]), axis = 1)
comp = ypred-test['AdoptionSpeed']
num_correct = len(np.where(comp == 0)[0])
print("Accuracy on testing set is: ", num_correct/len(test))

















