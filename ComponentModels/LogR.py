import numpy as np
import ComponentModel
from sklearn.linear_model import LogisticRegression

class LogR(ComponentModel.ComponentModel):
    def __init__(self,algo='lbfgs', c=100): #, OTHER PARAMS):
        self.regression = LogisticRegression(solver=algo,C=c)

    def train(self, x_train, y_train):
        self.regression.fit(x_train, y_train)

    def predict(self, x_test):
        return self.regression.predict(x_test)

    def report_accuracy(self, x_test, y_test):
        return self.regression.score(x_test, y_test)
