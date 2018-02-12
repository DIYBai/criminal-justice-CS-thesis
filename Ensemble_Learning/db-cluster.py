import numpy as np
import RiskParser as rp
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

#for timing purposes
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


#needed for breast cancer data
# from sklearn.preprocessing import StandardScaler
# in_file = "../Data/breast_cancer.csv"
# inputs, outputs = rp.parse_data(in_file, firstInpCol = 1, lastInpCol = 10)
# inputs = StandardScaler().fit_transform(inputs)

in_file = "../Data/RiskAssessData.csv"
inputs, outputs = rp.parse_data(in_file)

#below gotten from http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
db = DBSCAN(eps=1, min_samples=75).fit(inputs)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # Number of clusters in labels, ignoring noise if present.
# print(labels)
print("** DBSCAN **")
print( "Outliers:", np.count_nonzero(labels == -1) ) #https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
print('Estimated number of clusters: %d' % n_clusters_)


print("\n** K-Means **")
print("trial\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI")
#TODO: cluter w k-means
# below code from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'#\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_)#,
             # metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=sample_size)
            )
         )

for i in range(1, 30):
    kmeans = KMeans(init = 'k-means++', n_clusters = i, n_init = 20)
    bench_k_means(kmeans, name = ("k-means " + str(i)), data = inputs)
