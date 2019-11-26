from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from get_data_final import data
from mpl_toolkits.mplot3d import Axes3D
labels = data['AdoptionSpeed'].values
data = data.drop('AdoptionSpeed',axis=1)
data = data.values

# Let n denote number of features
n = len(data[0])

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


feats = np.array([1, 2, 3, 19, 20], dtype=int)
mask = np.zeros((1, n+1), dtype='bool')[0]
mask[feats] = 1


# Remove the label
x = data[:, :-1]


# PCA
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=3)
PC = pca.fit_transform(x)


# Plot the samples with different colors:
# 0: black
# 1: red
# 2: blue
# 3: green
# 4: yellow
fig1 = plt.figure()
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
plt.title('2D plot of PCA with all features')
black = mpatches.Patch(color='black', label='category 0')
red = mpatches.Patch(color='red', label='category 1')
blue = mpatches.Patch(color='blue', label='category 2')
green = mpatches.Patch(color='green', label='category 3')
yellow = mpatches.Patch(color='yellow', label='category 4')
plt.legend(handles=[black, red, blue, green, yellow])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# 3d Plot
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
mks = 0.50
plt.title('3D plot of PCA with all features')
ax.scatter(cate_0[:, 0], cate_0[:, 1], cate_0[:, 2], c='k', s=mks, marker='o')
ax.scatter(cate_1[:, 0], cate_1[:, 1], cate_1[:, 2], c='r', s=mks, marker='o')
ax.scatter(cate_2[:, 0], cate_2[:, 1], cate_2[:, 2], c='b', s=mks, marker='o')
ax.scatter(cate_3[:, 0], cate_3[:, 1], cate_3[:, 2], c='g', s=mks, marker='o')
ax.scatter(cate_4[:, 0], cate_4[:, 1], cate_4[:, 2], c='y', s=mks, marker='o')
black = mpatches.Patch(color='black', label='category 0')
red = mpatches.Patch(color='red', label='category 1')
blue = mpatches.Patch(color='blue', label='category 2')
green = mpatches.Patch(color='green', label='category 3')
yellow = mpatches.Patch(color='yellow', label='category 4')
plt.legend(handles=[black, red, blue, green, yellow])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


















