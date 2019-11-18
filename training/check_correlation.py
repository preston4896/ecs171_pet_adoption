import astetik as ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
newData = np.load("data_without_outliers.npy",allow_pickle=True)

newData = pd.DataFrame(newData, dtype=float)
corr = newData.corr(method='pearson')
newData.columns = ['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',
              'Maturity Size','Fur Length','Vaccinated','Dewormed','Sterilized',
              'Health','Quantity','Fee','State','Video Amount','Photo Amount',
              'Sentmt Magnitude','Sentmt Score','Adoption Speed']
ast.corr(newData)

plt.show()

