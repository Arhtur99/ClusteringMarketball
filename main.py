from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import random

import functions

random.seed(10)


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


player_data = pd.DataFrame(pd.read_csv("csv_marketball_players.csv"))
player_data.dropna(inplace=True, axis=0)



drop_cols = ['NUMBER', 'ROLE', 'MV (EUR)', 'PLAYER']
#print(player_data.columns)
player_data_prep = player_data.drop(drop_cols, axis=1)

categorical_columns = player_data_prep.select_dtypes(include=['object']).columns
numerical_columns = player_data_prep.select_dtypes(include=['float64', 'int64']).columns

# Dummy code categorical columns
dummy_coded_data = pd.get_dummies(player_data_prep, columns=categorical_columns)

# Scale numerical columns
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(player_data_prep[numerical_columns]), columns=numerical_columns)

# Combine scaled numerical columns and dummy coded columns
final_data = pd.concat([scaled_data, dummy_coded_data], axis=1)

# Remove duplicate columns
final_data = final_data.loc[:, ~final_data.columns.duplicated()]

# Define the pipeline
pipeline = Pipeline([
    ('clusterer', KMeans(n_clusters=5))  # Perform clustering with K-means
])

# Fit the pipeline to the final data
pipeline.fit(final_data)

# Get the cluster labels
clusters = pipeline['clusterer'].labels_


# Create a DataFrame with the "value" column and cluster labels
df = pd.DataFrame({'MV (EUR).1': player_data_prep['MV (EUR).1'], 'Cluster': clusters,
                   'PLAYER': player_data['PLAYER']})


clusters_df = pd.DataFrame(data=clusters, columns=['Cluster'])

player_w_clusters = pd.concat([clusters_df, player_data], axis=1)
print(player_w_clusters.columns)
#with pd.option_context('display.max_rows', None,
                       #'display.max_columns', None,
                       #'display.precision', 3,
                       #):
    #print(player_w_clusters.columns)

# Create the boxplot using matplotlib
plt.boxplot([df[df['Cluster'] == i]['MV (EUR).1'] for i in df['Cluster'].unique()],
            labels=['Cluster {}'.format(i) for i in df['Cluster'].unique()])
plt.xlabel('Cluster')
plt.ylabel('MV (EUR).1')
plt.title('Boxplot of Value by Cluster')
plt.show()


# Specify the column name representing the values and find outliers
outliers = functions.find_cluster_outliers(df)

# Print the outliers for each cluster
for cluster, outlier_values in outliers.items():
    print(f"Cluster {cluster} outliers: {outlier_values}")