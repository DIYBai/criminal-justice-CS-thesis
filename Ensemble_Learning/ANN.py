import numpy as np
import MLModel
from sklearn.neural_network import MLPClassifier

class ANN(MLModel.MLModel):
    def __init__(self): #, OTHER PARAMS):
        self.mlp = MLPClassifier( hidden_layer_sizes = (10,2,) )

    # def __init__(self, layer_sizes):
    #     self.mlp = MLPClassifier( hidden_layer_sizes = layer_sizes )

    def train(self, x_train, y_train):
        self.mlp.fit(x_train, y_train)

    def predict(self, x_test):
        return self.mlp.predict(x_test)

    def report_accuracy(self, x_test, y_test):
        return self.mlp.score(x_test, y_test)
