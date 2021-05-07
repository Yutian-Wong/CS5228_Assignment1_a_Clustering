import numpy as np
import matplotlib.pyplot as plt

from itertools import chain, combinations


def powerset(iterable, min_len=None, max_len=None):
    s = list(iterable)
    if min_len is None:
        min_len = 0
    if max_len is None:
        max_len = len(s)
    return chain.from_iterable(combinations(sorted(s), r) for r in range(min_len, max_len+1))


def binary_split(iterable):
    splits = []
    
    for X in powerset(iterable, min_len=1, max_len=len(iterable)-1):
        Y = tuple(sorted(set(iterable) - set(X)))
        yield (X, Y)
        
        
        
        
        
def plot_clusters(X, clusters, centroids):
    plt.figure()

    for cluster_id, cluster_samples in clusters.items():
        centroid = centroids[cluster_id]
        X_cluster = X[cluster_samples]
        if X_cluster.shape[0] > 0:
            plt.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color='C{}'.format(cluster_id), s=150)

            for x in X_cluster:
                plt.plot([x[0],centroid[0]], [x[1],centroid[1]], '--', linewidth=0.5, color='k'.format(cluster_id))
            
        plt.scatter(centroid[0], centroid[1], marker='+', color='k', s=250, lw=5)

    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)            
    plt.tight_layout()
    
    plt.show()        
    
    
    
def plot_labeled_data(X, labels, circle_coords=None, circle_radius=None, show_grid=False):
    plt.figure()
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    
    ax = plt.gca()
    
    ticks = np.arange(0, 10, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # Set the aspect ratio 1:1 (otherwise circles a stretched)
    plt.gca().set_aspect('equal')
    plt.scatter(X[:,0], X[:,1])
    if show_grid is True:
        plt.gca().yaxis.grid(color='gray', linestyle='dashed')
        plt.gca().xaxis.grid(color='gray', linestyle='dashed')
        plt.grid(True)
    # Add the labels to all points
    for i, label in enumerate(labels):
        plt.gca().annotate(label, X[i], fontsize=14)
    # Place the circle at the point of interest to easier evaluate the neighborhood of a point, if needed
    if circle_coords is not None and circle_radius is not None:
        circle = plt.Circle(circle_coords, circle_radius, color='b', fill=False)
        plt.gca().add_patch(circle)
    plt.show()
    
    
    
def calculate_sse(X, clusters, centroids):
    
    sse = 0
    
    ### Your code starts here ###############################################################
    
    for cluster_id in clusters:
        centroid = centroids[cluster_id]
        cluster_samples = X[clusters[cluster_id]]
        
        if len(cluster_samples) == 0:
            continue

        #distances = np.linalg.norm(cluster_samples - centroid, axis=1)
        sse_cluster = ((cluster_samples - centroid)**2).sum()
        
        sse += sse_cluster
    
    ### Your code ends here #################################################################
    
    return sse    
    