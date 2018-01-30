import numpy as np
import MLModel
from sklearn.linear_model import LogisticRegression

class Dtree(MLModel.MLModel):
    def __init__(self, k): #, OTHER PARAMS):
        self.regression = LogisticRegression(C=100)

    def train(self, x_train, y_train):
        self.regression.fit(self.x_train, self.y_train)

    def report_accuracy(self, x_text, y_test):
        return self.regression.score(self.x_test, self.y_test)
