import numpy as np
import MLModel
from sklearn.neural_network import MLPClassifier

class ANN(MLModel.MLModel):
    def __init__(self, algo='adam', activation_f='relu', learning_method='constant', iteration=200, learning_rate=.001, a=.0001,layers=[100]): #, OTHER PARAMS):
        self.mlp = MLPClassifier( solver= algo,activation=activation_f, max_iter=iteration ,learning_rate=learning_method,learning_rate_init=learning_rate, alpha= a,hidden_layer_sizes = layers )

    # def __init__(self, layer_sizes):
    #     self.mlp = MLPClassifier( hidden_layer_sizes = layer_sizes )

    def train(self, x_train, y_train):
        self.mlp.fit(x_train, y_train)

    def predict(self, x_test):
        return self.mlp.predict(x_test)

    def report_accuracy(self, x_test, y_test):
        return self.mlp.score(x_test, y_test)
