import numpy as np
import MLModel
from sklearn.linear_model import LogisticRegression

class LogR(MLModel.MLModel):
    def __init__(self): #, OTHER PARAMS):
        self.regression = LogisticRegression(C=100)

    def train(self, x_train, y_train):
        self.regression.fit(x_train, y_train)

    def predict(self, x_test):
        return self.regression.predict(x_test)

    def report_accuracy(self, x_test, y_test):
        return self.regression.score(x_test, y_test)
