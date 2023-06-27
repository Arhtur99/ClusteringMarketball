import numpy as np

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