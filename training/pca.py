from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
data = np.load('data_shuffled.npy', allow_pickle=True)
data = np.asarray(data)

# Let n denote number of features
n = len(data[0]) - 1
labels = data[:, n]

# change the column you want to include in PCS to 1
feature_names = np.array(['Type',                 #0
                          'Age',                  #1
                          'Breed1',               #2
                          'Breed2',               #3
                          'Gender',               #4
                          'Color1',               #5
                          'Color2',               #6
                          'Color3',               #7
                          'MaturitySize',         #8
                          'FurLength',            #9
                          'Vaccinated',           #10
                          'Dewormed',             #11
                          'Sterilized',           #12
                          'Health',               #13
                          'Quantity',             #14
                          'Fee',                  #15
                          'State',                #16
                          'VideoAmt',             #17
                          'PhotoAmt',             #18
                          'Magnitude',            #19
                          'Score',                #20
                          'AdoptionSpeed'],       #21
                           dtype=str)


feats = np.array([1, 2, 3, 20], dtype=int)
mask = np.zeros((1, n+1), dtype='bool')[0]
mask[:] = 1


# Remove the label
x = data[:, mask]


# PCA
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
PC = pca.fit_transform(x)


# Plot the samples with different colors:
# 0: black
# 1: red
# 2: blue
# 3: green
# 4: yellow
marker_size = 1.5
cate_0 = PC[np.where(labels == 0)[0], :]
cate_1 = PC[np.where(labels == 1)[0], :]
cate_2 = PC[np.where(labels == 2)[0], :]
cate_3 = PC[np.where(labels == 3)[0], :]
cate_4 = PC[np.where(labels == 4)[0], :]
plt.plot(cate_0[:, 0], cate_0[:, 1], 'k.', markersize=marker_size)
plt.plot(cate_1[:, 0], cate_1[:, 1], 'r.', markersize=marker_size)
plt.plot(cate_2[:, 0], cate_2[:, 1], 'b.', markersize=marker_size)
plt.plot(cate_3[:, 0], cate_3[:, 1], 'g.', markersize=marker_size)
plt.plot(cate_4[:, 0], cate_4[:, 1], 'y.', markersize=marker_size)


# Give labels and legends
plt.title('PCA with all features')
black = mpatches.Patch(color='black', label='category 0')
red = mpatches.Patch(color='red', label='category 1')
blue = mpatches.Patch(color='blue', label='category 2')
green = mpatches.Patch(color='green', label='category 3')
yellow = mpatches.Patch(color='yellow', label='category 4')
plt.legend(handles=[black, red, blue, green, yellow])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

