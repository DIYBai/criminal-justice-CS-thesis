import numpy as np
import MLModel
from sklearn.neighbors import KNeighborsClassifier

class KNN(MLModel.MLModel):
    def __init__(self, k): #, OTHER PARAMS):
        self.knn = KNeighborsClassifier(n_neighbors = k)

    def train(self, x_train, y_train):
        self.knn.fit(self.x_train, self.y_train)

    def report_accuracy(self, x_text, y_test):
        return self.knn.score(self.x_test, self.y_test)
