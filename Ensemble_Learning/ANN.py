import numpy as np
# import mglearn
from sklearn.neural_network import MLPClassifier

class ANN(MLModel):
    def __init__(self): #, OTHER PARAMS):
        self.mlp = MLPClassifier( hidden_layer_sizes = (10,2,) )

    # def __init__(self, layer_sizes):
    #     self.mlp = MLPClassifier( hidden_layer_sizes = layer_sizes )

    def train(self, x_train, y_train):
        self.mlp.fit(self.x_train, self.y_train)

    def report_accuracy(self, x_text, y_test):
        return self.mlp.score(self.x_test, self.y_test)
