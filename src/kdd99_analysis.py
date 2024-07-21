# Elbow Method for Determining Cluster Number 

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load Data from file
data = pd.read_csv('../data/raw/KDDCup99.csv')

# Dictionary to store LabelEncoders for each string column
encoders = {}

# Convert string data to numeric data
# It iterates through each column in the DataFrame and checks if the data type is 'object'.
# The LabelEncoder from scikit-learn is then used to transform strings such as 'tcp' into integers.
# Reason: K-Means algorithm requires numeric input
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le  # Store the encoder

# This line selects all columns except for the last one.
# The last column is excluded because it contains the label.
# We only need the input features (not the labels) because K-Means is an unsupervised algorithm.
# The `.values` at the end converts the DataFrame slice into a NumPy array, which is the preferred format for scikit-learn models.
X = data.iloc[:, :-1].values

# Value Scaling for K-Means
# Features with larger ranges could disproportionately influence the outcome.
# The StandardScaler standardizes the dataset by subtracting the mean and then scaling to unit variance.
# This means each feature will have a mean of 0 and a standard deviation of 1.
# The `fit_transform` method fits the scaler to the data and then transforms it in one step.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KElbowVisualizer to calculate the Elbow criterion and determine the optimal number of clusters
# plots the explained variance as a function of the number of clusters and looks for a kink in the curve
# that suggests the optimal number of clusters.
# KMeans is used as the clustering algorithm with settings from 1 to 15 clusters,
# and 'timings=True' enables the timing of the fit process for each k value.
visualizer = KElbowVisualizer(KMeans(random_state=42), k=(1,15), timings=True)
# Fit the data to the visualizer
visualizer.fit(X_scaled)
optimal_k = visualizer.elbow_value_

# If you like to visualize any plot, decomment the following two lines
# fig = plt.figure()
# fig.show()

# If you like to visualize the plots for data pre-clustering, decommment the following 3 lines
# df_pca = PCA(n_components=2).fit_transform(X_scaled)
# plt.scatter(df_pca[:,0], df_pca[:,1],alpha=1)
# fig.show()

# Perform K-Means clustering with the optimal number of clusters determined by the Elbow method
# Fit the K-Means model to the scaled data
# Extract the labels for each data point, indicating cluster membership
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled) 
labels = kmeans.labels_ 

# If you like to visualize the plot for data post-clustering, decomment the following three lines
# fig2 = plt.figure()
# plt.scatter(df_pca[:,0], df_pca[:,1], alpha=0.8, c=labels, cmap='jet')
# fig2.show()

# If you like to visualize any plot
# plt.show()

# Calculate distances to cluster centers
distances = kmeans.transform(X_scaled).min(axis=1)

# Set a threshold at the 95th percentile of the distance values to define anomalies
# Identify anomalies as data points where the distance exceeds the threshold
threshold = np.percentile(distances, 95)
anomalies = data[distances > threshold]

# Reverse the label encoding to convert numeric values back to their original string representations
for col, encoder in encoders.items():
    anomalies.loc[:, col] = encoder.inverse_transform(anomalies[col])

# Saving the anomalies in a CSV file
anomalies.to_csv('../results/anomalies.csv', index=False)

# Print Results
print(f"Optimal Number of Clusters: {optimal_k}")
print(f"Number of Anomalies: {anomalies.shape[0]}")
print("Anomalies were saved in 'anomalies.csv'.")
