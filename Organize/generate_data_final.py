import pandas as pd
import numpy as np
import json
from sklearn import metrics

# This file need sentiment folder and train.csv to run
# Read Data
data = pd.read_csv('train.csv')

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


# missing files replaced with average
averageMagnitude = magnitude / (len(data) - fileNotFound)
averageScore = score / (len(data) - fileNotFound)

data_temp[fileMissingIndex, -2] = averageMagnitude
data_temp[fileMissingIndex, -1] = averageScore

data['magnitude']=data_temp[:, -2]
data['score']=data_temp[:, -1]

data.to_pickle('final_data_frame')














