import numpy as np
import RiskParser as rp
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

#needed for breast cancer data
# from sklearn.preprocessing import StandardScaler
# in_file = "../Data/breast_cancer.csv"
# inputs, outputs = rp.parse_data(in_file, firstInpCol = 1, lastInpCol = 10)
# inputs = StandardScaler().fit_transform(inputs)

in_file = "../Data/RiskAssessData.csv"
inputs, outputs = rp.parse_data(in_file)

#below gotten from http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
db = DBSCAN(eps=0.5, min_samples=75).fit(inputs)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # Number of clusters in labels, ignoring noise if present.
# print(labels)
print( np.count_nonzero(labels == -1) ) #https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
print('Estimated number of clusters: %d' % n_clusters_)
