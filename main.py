import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


player_data = pd.DataFrame(pd.read_csv("csv_marketball_players.csv"))
player_data.dropna(inplace=True, axis=0)




drop_cols = ['NUMBER', 'ROLE', 'MV (EUR)', 'PLAYER']
#print(player_data.columns)
player_data_prep = player_data.drop(drop_cols, axis=1)

X = player_data.select_dtypes(include=numerics)
print(X)

scaler = StandardScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(X))

dummies = pd.get_dummies(player_data_prep)

n_clusters = 5
n_samples = 300
epochs = 100

kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=epochs)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
#plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
#plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
#plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
#plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
#Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.show()