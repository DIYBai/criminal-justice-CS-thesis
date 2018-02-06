import numpy as np
import MLModel
from sklearn.neighbors import KNeighborsClassifier

class KNN(MLModel.MLModel):
    def __init__(self, k=4 ,w='distance', algo='auto'): #, OTHER PARAMS):
        self.knn = KNeighborsClassifier(n_neighbors = k, weights =w, algorithm=algo)

    def train(self, x_train, y_train):
        self.knn.fit(x_train, y_train)

    def predict(self, x_test):
        #print "In KNN PREDICT"
    	return self.knn.predict(x_test)

    def report_accuracy(self, x_test, y_test):
        return self.knn.score(x_test, y_test)
