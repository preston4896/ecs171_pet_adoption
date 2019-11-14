import numpy as np
import pandas as pd
import json

data = np.load('data.npy', allow_pickle=True)
petID = pd.read_csv('train.csv', sep=',', usecols=['PetID']).values
zeros = np.zeros((len(data), 2), dtype=float)
data_temp = np.concatenate((data[:, 0:len(data[0])-1], zeros), 1)
print(data_temp)
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

data = np.concatenate((data_temp, np.array(data[:, -1], ndmin=2).T), 1)
data = np.random.shuffle(data)
print("Could not find {} files".format(fileNotFound))
data = np.save('data_shuffled', data, allow_pickle=True)
