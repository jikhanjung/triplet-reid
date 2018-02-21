# Authors: Mathew Kallada
# License: BSD 3 clause
"""
=========================================
Plot Hierarachical Clustering Dendrogram
=========================================
This example plots the corresponding dendrogram of a hierarchical clustering
using AgglomerativeClustering and the dendrogram method available in scipy.
"""

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def mpl_plot_embedding(pos,pids, args,  n_classes):
    dimension = pos.shape[1]
    n_samples = pos.shape[0]
    palette = sns.color_palette("hls", n_classes)

    # Iterator for different markers so differnet classes are clearly displayed.
    palette = sns.color_palette("hls", 9)
    markers = ["*", "+", "x", ">", "."]
    marker_iterator = (product(markers, palette))
    fig = plt.figure()
    if args.dimension == 2:
        ax = fig.add_subplot(111)
    elif args.dimension == 3:
        ax = Axes3D(fig)


    prev_pid = None
    for i in range(n_samples):
        pid = pids[i]
        if pid != prev_pid:
            print(prev_pid,  pid)
            marker, color = marker_iterator.next()
        if dimension == 2:
            x, y = pos[i, 0], pos[i, 1]
            ax.scatter(x, y, marker=marker, c=color)
        elif dimension == 3:
            x, y, z = pos[i]
            ax.scatter(x, y, z, c=color, marker=marker)

        prev_pid = pid

    if dimension == 2:
        fig.show()
    elif dimension == 3:
        # Animate rotating 3d axis
        angle = 0
        step = 3
        while True:
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)
            angle = (angle + 3) % 360

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


iris = load_iris()
x = iris.data[:20]
model = AgglomerativeClustering(n_clusters=3)

model = model.fit(x)
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(model, labels=model.labels_)
plt.show()