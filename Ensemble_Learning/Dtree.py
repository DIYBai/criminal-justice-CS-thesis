import numpy as np
import MLModel
from sklearn.tree import DecisionTreeClassifier

class Dtree(MLModel.MLModel):
    def __init__(self): #, OTHER PARAMS):
        self.dtree = DecisionTreeClassifier(random_state=0)

    def train(self, x_train, y_train):
        self.dtree.fit(x_train, y_train)

    def report_accuracy(self, x_test, y_test):
        return self.dtree.score(x_test, y_test)
