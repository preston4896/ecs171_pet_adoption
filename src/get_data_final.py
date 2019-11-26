import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from OutlierDetection import outlier_detection
np.random.seed(0)

# Read the data frame
data = pd.read_pickle('final_data_frame')

# Drop none sense features
data = data.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)

# 0.2 testing set, 0.8 training set
# Shuffle, with random_state set, everyone should use the same training and testing set
data = data.sample(frac=1, random_state=0)
ts = 0.2
ts = int(len(data)*ts)
train = data[:-ts]
train = outlier_detection(train)
test = data[-ts:]

# Create Variables needed
x_train = train.drop(columns='AdoptionSpeed')
y_train = train['AdoptionSpeed']

x_test = test.drop(columns='AdoptionSpeed')
y_test = test['AdoptionSpeed']

# Make sure every body use the same 
kf = KFold(n_splits=5, random_state=0, shuffle=True)

# Note: When doing 5-fold cross validation, use the following for loop to index access
#for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)





















