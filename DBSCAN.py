# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:29:10 2021

@author: Wang Yutian

CS5228_Assignment1a_Clustering 
Clustering algorithms: DBSCAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

# We provide you with some utility methods to plot the data
from utils import plot_clusters, plot_labeled_data, calculate_sse

df_dbscan = pd.read_csv('data/dbscan-demo-data.csv', header=None)

# Convert dataframe with coordinates to numpy array
X_dbscan = df_dbscan.to_numpy()
# Label all data points from 0 to (N-1), with N = #points
labels_dbscan = list(range(X_dbscan.shape[0]))
print(labels_dbscan)
print(X_dbscan.shape[0])


plot_labeled_data(X_dbscan, labels_dbscan, circle_coords=[6,7], circle_radius=1.25)


dbscan_core_points = [1,3,4,6,11]           # Example format: dbscan_core_points = [0, 1, 2, ...]

dbscan_border_points = [0,2,5,7,8,10,12,13]         # Example format: dbscan_border_points = [0, 1, 2, ...]

dbscan_noise_points = [9,14,15,16,17]          # Example format: dbscan_noise_points = [0, 1, 2, ...]


# Example format
#
# clustering_dbscan = [ 
#     [0, 1, 2, ...],
#     ...
# ]

clustering_dbscan = [
    [[1,3,4,6,0,2,5,7,8],
     [11,10,12,13]
    ]
]


# Example format
#
# X_dbscan_extra = [ [x1,y1], [x2, y2] ]


X_dbscan_extra = [[3, 5], [7, 6]]  # (keep the empty list if such 2 points do not exist)