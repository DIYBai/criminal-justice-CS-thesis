import numpy as np
import ComponentModel
from sklearn.neighbors import KNeighborsClassifier

class KNN(ComponentModel.ComponentModel):
    def __init__(self, k=4 ,w='distance', algo='auto'): #, OTHER PARAMS):
        self.knn = KNeighborsClassifier(n_neighbors = k, weights =w, algorithm=algo)

    def train(self, x_train, y_train):
        self.knn.fit(x_train, y_train)

    def predict(self, x_test):
        #print "In KNN PREDICT"
    	return self.knn.predict(x_test)

    def report_accuracy(self, x_test, y_test):
        return self.knn.score(x_test, y_test)

    def predict_prob(self, x_test):
        return self.knn.predict_proba(x_test)
