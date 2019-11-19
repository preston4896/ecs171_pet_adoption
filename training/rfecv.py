import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from get_data import *

df = pd.DataFrame(data)
df.columns = ['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',
              'Maturity Size','Fur Length','Vaccinated','Dewormed','Sterilized',
              'Health','Quantity','Fee','State','Video Amount','Photo Amount',
              'Sentmt Magnitude','Sentmt Score','Adoption Speed']
df['score*mag'] = df.apply(lambda row: (row['Sentmt Magnitude']*row['Sentmt Score']), axis=1)
df['Sterilized & Dewormed & Vacciniated'] = df.apply(lambda row: (row['Sterilized']*row['Dewormed']*row['Vaccinated']), axis=1)
df = df.drop('Dewormed',axis=1)
df = df.drop('Vaccinated',axis=1)
df = df.drop('Sterilized',axis=1)
X = df.drop('Adoption Speed', axis=1)
target = df['Adoption Speed']

label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)

#RFECV
rfc = RandomForestClassifier(n_estimators=100,n_jobs=-1,oob_score=True,max_features='auto',min_samples_leaf=50)
rfecv = RFECV(estimator=rfc, step=1, cv=3,  scoring='accuracy', verbose=1)
rfecv.fit(X, target)

print('Optimal number of features: {}'.format(rfecv.n_features_))
print( 'Least Important features: ', df.columns[np.where(rfecv.support_ == False)] )

#Drop the least important features
#This is the final data after RFECV.
X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
X['Adoption Speed'] = df['Adoption Speed']
np.save("new_data_shuffled",X)

# ===OPTIONAL MODEL FOR QUICK EVALUATION===

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import SGD
# from keras.optimizers import Adam
# from keras import regularizers
# from keras.callbacks import LambdaCallback
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=42, shuffle=True)
# y_train_encoded = to_categorical(y_train, 5)
# y_test_encoded  = to_categorical(y_test,  5)

# model = Sequential()
# model.add(Dense(10, input_dim=X.shape[1], activation='sigmoid', kernel_initializer='random_uniform'))
# model.add(Dense(10, activation='sigmoid', kernel_initializer='random_uniform'))
# model.add(Dense(5, activation='softmax', kernel_initializer='random_uniform',))
# model.compile(optimizer=Adam(learning_rate=0.1), loss='mse', metrics=['accuracy'])

# # Train
# eval_acc = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.evaluate(X_test, y_test_encoded)[1]))
# model.fit(X_train, y_train
# # Evaluate
# print('Evaluation:', model.evaluate(X_test, y_test_encoded)[1])