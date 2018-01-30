import numpy as np
import MLModel
from sklearn.tree import DecisionTreeClassifier

class Dtree(MLModel.MLModel):
    def __init__(self, k): #, OTHER PARAMS):
        self.dtree = DecisionTreeClassifier(random_state=0)

    def train(self, x_train, y_train):
        self.dtree.fit(self.x_train, self.y_train)

    def predict(self, x_test):
    	return self.dtree.predict(x_test)

    def report_accuracy(self, x_text, y_test):
        return self.dtree.score(self.x_test, self.y_test)
