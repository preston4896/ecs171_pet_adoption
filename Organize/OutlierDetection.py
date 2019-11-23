import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def outlier_detection(df):
    X = df.drop('AdoptionSpeed', axis=1)
    Y = df['AdoptionSpeed']

    # Isolation Forest
    clf = IsolationForest(contamination=0.08, max_samples=256, behaviour="new") # 8% outliers
    clf.fit(X)
    processedData1 = clf.predict(X) == -1

    print("Isolation Forest: ", sum(processedData1))

    clf = LocalOutlierFactor(contamination='auto')
    processedData2 = clf.fit_predict(X) == -1
    print("LOF:", sum(processedData2))

    newData = df.loc[processedData1 == 0]
    newData = newData.reset_index(drop=True)

    return newData

