import numpy as np
import EnsembleModel
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
lib_path = os.path.abspath(os.path.join('..','ComponentModels'))
sys.path.append(lib_path)
from ANN import ANN
import copy
import numpy as np

class fullSL(EnsembleModel.EnsembleModel):

    ##  takes in list of component models  ##
    def __init__(self, model_list):
        self.model_list = model_list
        self.model_accuracies = []
        self.best_model = model_list[0]
        self.total_accuracy = 100.000

    ##  creates deep copy of dataset, with ith fold removed  ##
    def concatenate(self, folds, i):
        folds_copy = copy.deepcopy(folds)
        test = folds_copy.pop(i)
        return np.concatenate(folds_copy, axis=0), test

    ##  gets overall accuracies for each model, and total accuracy  ##
    def mean_accuracies(self,accuracies_list,b,n=5):
        self.model_accuracies = [sum(x)/b for x in zip(*accuracies_list)]
        model_accuracies_np = np.asarray(self.model_accuracies)
        model_accuracies_np = [x**n for x in model_accuracies_np]
        self.total_accuracy = np.sum(model_accuracies_np)

    ##  splits the dataset into b folds, and trains component  ##
    ##  models on each fold while keeping track of accuracy  ##
    def train(self, inputs, outputs, b=10):
        input_folds = np.array_split(inputs, b)
        output_folds = np.array_split(outputs, b)
        accuracies_list = []
        for i in range(len(input_folds)):
            x_train, x_test = self.concatenate(input_folds, i)
            y_train, y_test = self.concatenate(output_folds, i)
            self.sub_model_train(x_train, y_train)
            accuracies = self.fold_accuracy(x_test,y_test)
            accuracies_list.append(accuracies)
        self.mean_accuracies(accuracies_list,b)
        self.sub_model_train(inputs,outputs)
   
    ##  trains componenet models on inputs  ##
    def sub_model_train(self, x_train, y_train):
        for i in range(len(self.model_list)):
            self.model_list[i].train(x_train,y_train)

    ##  gets prediciton on x input by weighting the votes from  ##
    ##  each component models with their corresponding accuracies  ##
    def predict(self, x_test):
        total_accuracy = 0.0
        accuracies = []
        for i in range(len(self.model_list)):
            accuracies.append(self.model_list[i].predict(x_test))
            total_accuracy += (accuracies[i] * self.model_accuracies[i])
        if total_accuracy >= (self.total_accuracy/2): return 1, accuracies
        return 0, accuracies

    ##  gets accuracy for each model in input ##
    def fold_accuracy(self, x_test, y_test):
        accuracies = []
        for i in range(len(self.model_list)):
            prediction = self.model_list[i].report_accuracy(x_test, y_test)
            accuracies.append(prediction)
        return accuracies

    ##  verifies prediction for each entry, and returns accuracy (correct/total)  ##
    def report_accuracy(self, x_test, y_test):
        count = 0
        for i in range(0, len(y_test)):
            prediction, accuracies = self.predict([x_test[i]])
            if prediction == y_test[i]:
               count += 1
            else:
               self.print_prediction(prediction, y_test[i], accuracies)
        return float(count)/float(len(y_test))

    ##  prints predicition for user  ##
    def print_prediction(self, predict, actual, accuracies):
        print("Prediction: " + str(predict) + " Actual: " + str(actual))
        for i in range(len(accuracies)):
            print("---> " + str(self.model_list[i]) + " Prediction: " + str(accuracies[i]) )