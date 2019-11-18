import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import astetik as ast
import seaborn as sns
import matplotlib.pyplot as plt
df = np.load('data_shuffled.npy', allow_pickle=True)

df= pd.DataFrame(df, dtype=float)
df.columns = ['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',
              'Maturity Size','Fur Length','Vaccinated','Dewormed','Sterilized',
              'Health','Quantity','Fee','State','Video Amount','Photo Amount',
              'Sentmt Magnitude','Sentmt Score','Adoption Speed']
X = df.drop('Adoption Speed', axis=1)
Y = df['Adoption Speed']

# Isolation Forest
clf = IsolationForest(contamination="auto", behaviour="new")
clf.fit(X)
processedData1 = clf.predict(X) == -1

print("Isolation Forest: ", sum(processedData1))

clf = LocalOutlierFactor(contamination='auto')
processedData2 = clf.fit_predict(X) == -1
print("LOF:", sum(processedData2))

newData = df.loc[processedData2 == 0]
newData = newData.reset_index(drop=True)





np.save("data_without_outliers",newData)
