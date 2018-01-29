import numpy as np
# import mglearn
from sklearn.tree import DecisionTreeClassifier

class Dtree(MLModel):
    def __init__(self, k): #, OTHER PARAMS):
        self.dtree = DecisionTreeClassifier(random_state=0)

    def train(self, x_train, y_train):
        self.dtree.fit(self.x_train, self.y_train)

    def report_accuracy(self, x_text, y_test):
        return self.dtree.score(self.x_test, self.y_test)
