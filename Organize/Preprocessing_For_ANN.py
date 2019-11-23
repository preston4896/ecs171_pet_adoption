import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
import numpy as np
from check_correlation import check_correlation
from OutlierDetection import outlier_detection
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from get_data_final import x_train,x_test,y_train,y_test
np.random.seed(0)

# Read the data frame
df = pd.read_pickle('final_data_frame')

# Drop none sense features
df = df.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)

# 0.2 testing set, 0.8 training set
# Shuffle, with random_state set, everyone should use the same training and testing set
df = df.sample(frac=1, random_state=0)
# Create Variables needed
check_correlation(x_train)
X = x_train
target = y_train
temp = X.values
scaler=MinMaxScaler()
temp = scaler.fit_transform(temp)
X = pd.DataFrame(temp, columns=X.columns)
temp = x_test.values
temp = scaler.fit_transform(temp)
x_test = pd.DataFrame(temp, columns=X.columns)
X['score*mag'] = X.apply(lambda row: (row['magnitude']*row['score']), axis=1)
x_test['score*mag'] = x_test.apply(lambda row: (row['magnitude']*row['score']), axis=1)
X['Dewormed*Vaccinated'] = X.apply(lambda row: (row['Dewormed']+row['Vaccinated'] - row['Dewormed']*row['Vaccinated']), axis=1)
X['Dewormed*Vaccinated'][X['Dewormed*Vaccinated'] == 0.75] = 0.5
x_test['Dewormed*Vaccinated'] = X.apply(lambda row: (row['Dewormed']+row['Vaccinated'] - row['Dewormed']*row['Vaccinated']), axis=1)
x_test['Dewormed*Vaccinated'][x_test['Dewormed*Vaccinated'] == 0.75] = 0.5
X = X.drop('Dewormed', axis=1)
X = X.drop('Vaccinated', axis=1)
x_test = x_test.drop('Dewormed', axis=1)
x_test = x_test.drop('Vaccinated', axis=1)

label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)

# RFECV
rfc = RandomForestClassifier(n_estimators=100,oob_score=True,max_features='auto',min_samples_leaf=50)
rfecv = RFECV(estimator=rfc, step=1, cv=5,  scoring='accuracy', verbose=1)
rfecv.fit(X, target)

print('Optimal number of features: {}'.format(rfecv.n_features_))
print( 'Least Important features: ', X.columns[np.where(rfecv.support_ == False)])

#Drop the least important features
#This is the final data after RFECV.
X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
x_test.drop(x_test.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)


n = X.shape[1]
x_train = X.values
x_test = x_test.values
y_train = to_categorical(y_train,5,dtype=int)
y_test = to_categorical(y_test,5,dtype=int)

