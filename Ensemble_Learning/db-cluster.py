import numpy as np
import RiskParser as rp
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

in_file = "../Data/breast_cancer.csv"
inputs, outputs = rp.parse_data(in_file, firstInpCol = 1, lastInpCol = 10)

#below gotten from http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
inputs = StandardScaler().fit_transform(inputs)
db = DBSCAN(eps=0.3, min_samples=5).fit(inputs)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # Number of clusters in labels, ignoring noise if present.
print(labels)
print('Estimated number of clusters: %d' % n_clusters_)
