# This lab will explore hierarchal linking, using a complete linkage strategy
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import scipy
import pylab
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import fcluster


# Step 1: Read in Data
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv"

df = pd.read_csv(path)

# Step 2: Data Wrangling
# For simplicity, kets drop rows that have non-numeric values.
df[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')

df.dropna(inplace = True)
df.reset_index(drop=True, inplace=True)

# Step 3: Get Feature Set
featureset = df[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# Step 4: Normalize.
# We will normalize the data using a MinMax Scaler, which will bring the values between 0 and 1
# The Standard scaler uses z-scores, which is based on the normal distribution
X = featureset.values
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(X)


# Step 5: Get the Distance Matrix and make the model with Scikit learn
dist_matrix = distance_matrix(feature_mtx,feature_mtx)
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)

# Step 6: Add the labels to the df as a feature
labels = agglom.labels_
df["cluster_"] = labels

# Step 7: Visualize what we've made
n_clusters = max(agglom.labels_) + 1
colors = cm.rainbow(np.linspace(0, 1, n_clusters)) # Makes the colors randomly
cluster_labels = list(range(0, n_clusters)) # makes a list of cluster labels from 0 to max-1. Labels are ints

plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = df[df.cluster_ == label]
    for i in subset.index:
        plt.text(subset.horsepow[i], subset.mpg[i], str(subset['model'][i]), rotation=25)

    plt.scatter(subset.horsepow, subset.mpg, s=subset.price * 10, c=color, label='cluster' + str(label), alpha=0.5)

plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()

# this graph is really tough as we can't really distinguish between the types of cars here
# Or see the centroids of the clusters
print(df.groupby(['cluster_','type'])['cluster_'].count())

agg_cars = df.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()

print(agg_cars)

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),] # gets the data for each cluster

    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()

# So there are six clusters
# Type=0 is a car, so there is cheap car, better milage, lower horsepower
# moderate car, moderate milage, moderate horsepower
# expensive car, low milage, high horsepower
# For trucks, the clisters are similar with a tighter spread.

# We can do something similar with scipy

# Step 1: Get the distance matrix
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
Z = hierarchy.linkage(D, 'complete')

# This basically uses a cutting line for max distance between clusters
# max_d = 3

# clusters = fcluster(Z, max_d, criterion='distance')

# alternatively, you can calculate the clusters directly using an explicit k val
k=5
clusters = fcluster(Z, k, criterion='maxclust')

fig = pylab.figure(figsize=(18, 50))

# This function is used to label the leaves in the dendrogram
def llf(id):
    return '[%s %s %s]' % (df['manufact'][id], df['model'][id], int(float(df['type'][id])))


dendro = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')
plt.show()
# Same plot as before









