import numpy as np
import MLModel
from sklearn.tree import DecisionTreeClassifier

class Dtree(MLModel.MLModel):
    def __init__(self, criteria, split , depth): #, OTHER PARAMS):
        self.dtree = DecisionTreeClassifier(criterion = criteria, splitter = split, max_depth=depth,random_state=0)

    def train(self, x_train, y_train):
        self.dtree.fit(x_train, y_train)

    def predict(self, x_test):
    	return self.dtree.predict(x_test)

    def report_accuracy(self, x_test, y_test):
        return self.dtree.score(x_test, y_test)
