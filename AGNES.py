# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:32:35 2021

@author: Wang Yutian

CS5228_Assignment1a_Clustering 
Clustering algorithms: AGNES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

# We provide you with some utility methods to plot the data
from utils import plot_clusters, plot_labeled_data, calculate_sse

df_agnes = pd.read_csv('data/agnes-demo-data.csv', header=None)

# This dataset has the data points labeled with capital letters
labels_agnes = df_agnes[0].to_list()
# Convert coordinate columns of the dataframe with coordinates to numpy array
X_agnes = df_agnes[[1,2]].to_numpy()

plot_labeled_data(X_agnes, labels_agnes, show_grid=True)

"""
In the following 2 tasks, perform Hierarchical Clustering (AGNES) step by 
step on the dataset above. After each step write down the current set of 
clusters and the value of the shortest distance!

Denote a clusters as a string of the labels of the points forming a cluster. 
For example, XYZ denotes the cluster containing the data points labeled X, Y, 
and Z -- the order is not important, i.e., clustering ['X', 'YZ'] is equal to ['ZY', 'X']

For each step, write down the shortest distance between the two clusters merged 
in each step. Note that the points are conveniently placed to make the 
calculation of distances pretty straightforward. If needed, round the distances 
to 2 decimal places (e.g., sqrt(2) = 1.41)

Below you can see an example:

"""
# Example Format
#
# agnes_clusters = [
#     (None, ['X', 'Y', 'Z']),   # At the start, each data point forms a cluster
#     (1.22, ['Y', 'XZ']),       # Cluster X and Z where closest with a distance of 1.22
#     (2.00, ['XYZ']),           # Cluster XZ and y where closest with a distance of 2.0
#     (None, ['XYZ'])            # At the end, all data points are within a single cluster
# ]

agnes_clusters_single = [
    (None, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']), # At start, each data point forms a cluster
    (1,['AD','B', 'C', 'E', 'F', 'G', 'H', 'I']),
    (1.41,['AD', 'FH','B', 'C', 'E', 'G', 'I']),
    (2,['AD', 'FH','BC', 'E', 'G', 'I']),
    (2.24,['AD', 'FH','BC' ,'E','GI']),
    (2.83,['ADBC', 'FH', 'E', 'GI']),
    (3,['ADBC', 'FHE', 'GI']),
    (3.16,['ADBCFHE', 'GI']),
    (4.12,['ADBCFHEGI']),
    (None, ['ABCDEFGHI'])       # At the end, all data points are within a single cluster
]

agnes_clusters_complete = [
    (None, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']), # At start, each data point forms a cluster
    (1,['AD','B', 'C', 'E', 'F', 'G', 'H', 'I']),
    (1.41,['AD', 'FH','B', 'C', 'E', 'G', 'I']),
    (2,['AD', 'FH','BC', 'E', 'G', 'I']),
    (2.24,['AD', 'FH','BC' ,'E','GI']),
    (3.61,['ADE', 'FH','BC', 'GI']),
    (5.83,['ADE', 'FH','BCGI']),
    (7.62,['ADEFH','BCGI']),
    (9.22,['ADEFHBCGI']),
    (None, ['ABCDEFGHI'])       # At the end, all data points are within a single cluster
]
