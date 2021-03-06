# -*- coding: utf-8 -*-
"""
Editor: Wang Yutian

CS5228_Assignment1a_Clustering 
Clustering algorithms: KMeans
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

# Some utility methods to plot the data
from utils import plot_clusters, plot_labeled_data, calculate_sse

plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.dpi'] = 100

X_kmeans, _ = make_blobs(n_samples=30, centers=3, n_features=2, cluster_std=0.85, random_state=11)

plt.figure()
plt.scatter(X_kmeans[:,0], X_kmeans[:,1])
plt.show()

k = 5

#Define initial centroids
def init_centroids(X, k):
    centroids = None
    
    num_samples, num_dimensions = X.shape
    Range_Max = X.max(axis = 0)
    Range_Min = X.min(axis = 0)
    centroids = np.random.rand(k,num_dimensions)
    for i in range(num_dimensions):
        centroids[:,i] = Range_Min[i] + centroids[:,i] * (Range_Max[i] - Range_Min[i])
    
    return centroids


#
# Example format:
#
# centroids = [ [2.35, -4.02 ], [0.64, 9.77 ], [ -3.50, 2,91 ] ]
#
centroids = init_centroids(X_kmeans, k)
print(centroids)
print(X_kmeans[0])
print(range(k))


"""
Calculating distances and finding the smallest values

The method assign_clusters() should return a dictionary 
where the keys represent the ids of the k cluster ranging 
from 0 to k-1, and each value is a list of indices of the 
data points belonging to the respective cluster
"""
def assign_clusters(X, k, centroids):
    # Reset all clusters
    clusters = {key: [] for key in range(k)}

    for idx, x in enumerate(X):
        
        cluster_id = 0
        Temp_array = np.zeros(k)
        for i in range(k):
            Temp_array[i] = np.linalg.norm(x - centroids[i])
            pass
        cluster_id = np.argmin(Temp_array)
        clusters[cluster_id].append(idx)
        
        # Only there so the empty loop does not throw an error
        pass

    return clusters

#
# Example format
#
#clusters = {
#   0: [2, 4, 6, 8],
#   1: [0, 1, 5],
#   2: [3, 7, 9]
#}
clusters = assign_clusters(X_kmeans, k, centroids)
print(clusters)


"""
Update Centroids

After the assignment of the data points to clusters, all
centroids need to be moved to the average of their
respective clusters. Note that the centroids might not 
change because the assignment made no changes to the 
clusters and K-Means is done. But we don't have to worry about that here.
"""
def update_centroids(X, clusters, centroids):
    
    new_centroids = np.zeros_like(centroids)

    for ithcluster in range(len(centroids)):
        if(np.size(clusters[ithcluster]) != 0 ):
            Temp_centroid = np.mean(X[clusters[ithcluster]], axis = 0)
            new_centroids[ithcluster] = Temp_centroid
        else:
            new_centroids[ithcluster] = centroids[ithcluster]
        pass
    
    return new_centroids


# Example format:
#
# centroids = [ [2.35, -4.02 ], [0.64, 9.77 ], [ -3.50, 2,91 ] ]
#
centroids = update_centroids(X_kmeans, clusters, centroids)
print(centroids)

"""
Putting all the functions together

Combine all three steps into a single method to perform 
K-Means over a dataset X given a choice for k.
The method kmeans should return a clustering 
(same format as assign_clusters() and update_centroids), 
as well as the list of centroids after convergence 
(same format as init_centroids()).
"""
def kmeans(X, k, max_iter=100, verbose=False):
    clusters, num_iterations = {}, 0
    
    centroids = init_centroids(X, k)

    for _ in range(max_iter):
        # Update the counter (+1 since we start from 0)
        num_iterations = num_iterations + 1
        
    clusters = assign_clusters(X, k, centroids)
    centroids = update_centroids(X, clusters, centroids)
    
    # Let's print the number of comparison
    if verbose is True:
        print('K-Means required {} iterations to converge.'.format(num_iterations))
    
    return clusters, centroids

#
# Example format
#
# clusters = {
#    0: [2, 4, 6, 8],
#    1: [0, 1, 5],
#    2: [3, 7, 9]
# }
#
# centroids = [ [2.35, -4.02 ], [0.64, 9.77 ], [ -3.50, 2,91 ] ]

"""
Test Implementation

Run your implementation of K-Means over the generated dataset 
and visualize the results using the plot_clusters method we provide for you.
This allows you to check if your implementation is (seemingly) correct -- 
important: note that an unlucky initialization might yield poor result, 
so you might run K-Means multiple times before you pass judgment on 
you implementation.
"""

clusters, centroids = kmeans(X_kmeans, 3)
print(clusters)
print(centroids)
print(calculate_sse(X_kmeans, clusters, centroids))

plot_clusters(X_kmeans, clusters, centroids)


"""
Parameter Tuning(Just Selecting a Good Value for k)

"""
df_kmeans = pd.read_csv('data/kmeans-demo-data.csv', header=None)

X_kmeans_highdim = df_kmeans.to_numpy()

num_samples, num_dimensions = X_kmeans_highdim.shape

print('The sample dataset has {} data points; each data point has {} attributes.'.format(num_samples, num_dimensions))

"""
Implement the method evaluate_k() that returns the SSE values for differen 
choices of k ranging from 2 to max_k for a dataset X. We provide you with 
the method calculate_sse() to calculate the SSE for clustering.
"""
def evaluate_k(X, max_k=30):

    sse_values = []

    for k in range (2, max_k+1):
        
        #
        # Note that the SSE value for each k will depend on the initialization.
        # You should there run kmeans2 multiple times (e.g., 10) and record only the lowest sse value
        #
        temp_sse = float('inf')
        for i in range(1, 40):
            clusters, centroids = kmeans(X, k)
            sse_cluster = calculate_sse(X,clusters, centroids)
            if( sse_cluster< temp_sse):
                temp_sse = sse_cluster
            i = i +1
        sse_values.append([k,temp_sse])
        
        # Only there so the empty loop does not throw an error
        # (you can remove that once you added your code)
        pass
        
    # Convert to numpy array to make the plotting easier 
    np.set_printoptions(suppress=True)
    sse_values = np.array(sse_values)
    
    return sse_values


# sse_values should contain the lowest SSE value for each k
# 
# Example format: sse_values = [ [2, 1000], [3, 340], [4, 121], ... ]
#
sse_values = evaluate_k(X_kmeans_highdim)
print(sse_values)

plt.figure()
plt.plot(sse_values[:,0], sse_values[:,1])
plt.tight_layout()
plt.show()


