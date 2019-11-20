import astetik as ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

newData = np.load("data_without_outliers.npy",allow_pickle=True)

newData = pd.DataFrame(newData, dtype=float)
newData.columns = ['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',
              'Maturity Size','Fur Length','Vaccinated','Dewormed','Sterilized',
              'Health','Quantity','Fee','State','Video Amount','Photo Amount',
              'Sentmt Magnitude','Sentmt Score','Adoption Speed']

ast.corr(newData, corr_method='pearson')


plt.show()

contingency_table = np.zeros((3,3),dtype=int)
contingency_table[0,0] = sum((newData['Vaccinated'] == 1) == (newData['Dewormed'] == 1))
contingency_table[0,1] = sum((newData['Vaccinated'] == 1) == (newData['Dewormed'] == 0))
contingency_table[0,2] = sum((newData['Vaccinated'] == 1) == (newData['Dewormed'] == 0.5))
contingency_table[1,0] = sum((newData['Vaccinated'] == 0) == (newData['Dewormed'] == 1))
contingency_table[1,1] = sum((newData['Vaccinated'] == 0) == (newData['Dewormed'] == 0))
contingency_table[1,2] = sum((newData['Vaccinated'] == 0) == (newData['Dewormed'] == 0.5))
contingency_table[2,0] = sum((newData['Vaccinated'] == 0.5) == (newData['Dewormed'] == 1))
contingency_table[2,1] = sum((newData['Vaccinated'] == 0.5) == (newData['Dewormed'] == 0))
contingency_table[2,2] = sum((newData['Vaccinated'] == 0.5) == (newData['Dewormed'] == 0.5))

stat, p, dof, expected = chi2_contingency(contingency_table)
prob = 0.95
# interpret p-value
alpha = 1.0 - prob
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')