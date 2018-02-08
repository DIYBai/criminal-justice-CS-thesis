import numpy as np
import ComponentModel
from sklearn.tree import DecisionTreeClassifier

class Dtree(ComponentModel.ComponentModel):
    def __init__(self, criteria='gini', split='best' , depth=40): #, OTHER PARAMS):
        self.dtree = DecisionTreeClassifier(criterion = criteria, splitter = split, max_depth=depth,random_state=0)

    def train(self, x_train, y_train):
        self.dtree.fit(x_train, y_train)

    def predict(self, x_test):
    	return self.dtree.predict(x_test)

    def report_accuracy(self, x_test, y_test):
        return self.dtree.score(x_test, y_test)
