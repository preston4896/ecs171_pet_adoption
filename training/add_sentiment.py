import numpy as np
import pandas as pd
import json

data = np.load('data.npy', allow_pickle=True)
petID = pd.read_csv('train.csv', sep=',', usecols=['PetID']).values
zeros = np.zeros((len(data), 2), dtype=float)
data = np.concatenate((data, zeros), 1)

fileNotFound = 0
fileMissingIndex = []
magnitude = 0
score = 0
for i in range(0, len(data)):
	try:
		fileName = 'train_sentiment/' + petID[i][0] + '.json'
		with open(fileName) as json_file:
		    info = json.load(json_file)
		    data[i, -2] = info['documentSentiment']['magnitude']
		    data[i, -1] = info['documentSentiment']['score']
		    magnitude+=info['documentSentiment']['magnitude']
		    score+=info['documentSentiment']['score']
	except:
		fileMissingIndex.append(i)
		fileNotFound += 1

		# uncomment to see which files are missing
		#print("Could not find {}.json".format(petID[i][0]))
		
averageMagnitude = magnitude / (len(data) - fileNotFound)
averageScore = score / (len(data) - fileNotFound)

data[fileMissingIndex, -2] = averageMagnitude
data[fileMissingIndex, -1] = averageScore
print("Could not find {} files".format(fileNotFound))
data = np.save('data', data, allow_pickle=True)
