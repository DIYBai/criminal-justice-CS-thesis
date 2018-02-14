import numpy as np
import EnsembleModel
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats
lib_path = os.path.abspath(os.path.join('..','ComponentModels'))
sys.path.append(lib_path)
from ANN import ANN
import copy
import numpy as np

class ensembleTester():

    def __init__(self, models, output_file = "model_results.txt"):
        self.models = models
        self.output_file = output_file

    def get_metrics(self,predictions,outputs):
        c_matrix = confusion_matrix(predictions,outputs)
        a_score = accuracy_score(predictions,outputs)
        print(c_matrix)
        print(a_score)
        return a_score,c_matrix

    
    def test_sample(self,inputs,outputs):
        accuracies = np.asarray([])
        matricies = np.asarray([])
        for i in range(0,len(self.models)):
            self.models[i].train(inputs, outputs)
            predictions = self.models[i].predict(inputs)
            a_score, c_matrix = self.get_metrics(predictions,outputs)
            np.append(accuracies, a_score)
            np.append(matricies, c_matrix)
        print("yes")

    def test(self,inputs,outputs,b=10):
        parser = ANN()
        for i in range(b):
            x_train, x_test, y_train, y_test = parser.split_data(inputs, outputs, .25)
            self.test_sample(x_train, y_train)
            self.test_sample(x_test, y_test) 
