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

random.seed(10)

def find_cluster_outliers(data):
    """
    Find outliers for each cluster in the data.

    Parameters:
    - data: pandas DataFrame containing the data
    - cluster_labels: pandas Series or array-like object containing the cluster labels
    - value_column: column name representing the values to analyze for outliers

    Returns:
    - Dictionary with cluster labels as keys and a list of outlier values as values
    """
    # Create a dictionary to store outlier values for each cluster
    outliers = {}
    cluster_labels = data['Cluster']
    value_column = data['MV (EUR).1']
    player_column = data['PLAYER']
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        # Select values for the current cluster
        cluster_values = data[data['Cluster'] == cluster]['MV (EUR).1']
        cluster_players = data[data['Cluster'] == cluster]['PLAYER']

        # Calculate quartiles and interquartile range (IQR)
        q1 = np.percentile(cluster_values, 25)
        q3 = np.percentile(cluster_values, 75)
        iqr = q3 - q1

        # Define the upper and lower bounds for outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Find outliers for the current cluster
        cluster_outliers = []
        for value, player in zip(cluster_values, cluster_players):
            if value < lower_bound or value > upper_bound:
                outlier_dict = {"player": player, "MV (EUR).1": value}
                cluster_outliers.append(outlier_dict)

        # Store outliers in the dictionary
        outliers[cluster] = cluster_outliers

    return outliers


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



big_df = pd.concat([df, player_data_prep], axis=1)


# Create the boxplot using matplotlib
plt.boxplot([df[df['Cluster'] == i]['MV (EUR).1'] for i in df['Cluster'].unique()],
            labels=['Cluster {}'.format(i) for i in df['Cluster'].unique()])
plt.xlabel('Cluster')
plt.ylabel('MV (EUR).1')
plt.title('Boxplot of Value by Cluster')
plt.show()


# Specify the column name representing the values and find outliers
outliers = find_cluster_outliers(df)

# Print the outliers for each cluster
for cluster, outlier_values in outliers.items():
    print(f"Cluster {cluster} outliers: {outlier_values}")