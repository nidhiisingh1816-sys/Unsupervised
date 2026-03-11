import numpy as np
from sklearn.cluster import KMeans

# Sample data
X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
])

# Create model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Train model
kmeans.fit(X)

# Get cluster labels
print("Cluster Labels:", kmeans.labels_)

# Predict cluster for new data
prediction = kmeans.predict([[0, 0]])
print("Cluster for [0,0]:", prediction)