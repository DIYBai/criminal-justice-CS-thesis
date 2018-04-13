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

##export probability instead
##use higher degree multiplication

class annSL(EnsembleModel.EnsembleModel):

    ## takes model list as an input  ##
    def __init__(self, model_list):
        self.model_list = model_list
        self.model_accuracies = []
        self.best_model = model_list[0]
        self.total_accuracy = 100.000
        self.meta_Alg = ANN(iteration=10000)

    ##  creates deep copy of dataset, with ith fold removed  ##
    def concatenate(self, folds, i):
        folds_copy = copy.deepcopy(folds)
        test = folds_copy.pop(i)
        return np.concatenate(folds_copy, axis=0), test

    ##  creates new matrix with prediciton marix and inputs  ##
    def concatenate_matrix(self, prediction_matrix, inputs):
        new_matrix = []
        for i in range(len(inputs)):
            new_arr = np.concatenate((prediction_matrix[i],inputs[i]))
            new_matrix.append(new_arr)
        return new_matrix

    ##  gets overall accuracies for each model, and total accuracy  ##
    def mean_accuracies(self,accuracies_list,b):
        self.model_accuracies = [sum(x)/b for x in zip(*accuracies_list)]
        accuracies_list_np = np.asarray(accuracies_list)
        self.total_accuracy = np.sum(accuracies_list_np)

    ##  splits the dataset into b folds, and trains component  ##
    ##  models on each fold while keeping track of accuracy  ##
    def train_components(self, inputs,outputs,b):
        input_folds = np.array_split(inputs, b)
        output_folds = np.array_split(outputs, b)
        accuracies_list = []
        for i in range(len(input_folds)):
            x_train, x_test = self.concatenate(input_folds, i)
            y_train, y_test = self.concatenate(output_folds, i)
            self.fold_train(x_train, y_train)
            accuracies = self.fold_accuracy(x_test,y_test)
            accuracies_list.append(accuracies)
        self.mean_accuracies(accuracies_list,b)
        ##print(self.model_accuracies)
        ##print(self.best_model)

    ##  trains each component model and gets predictions, then trains  ##
    ##  meta algorithm using these outputs and the actual outputs  ##
    def train(self, inputs, outputs, b=10, epochs=1000):
        self.train_components(inputs,outputs,b)
        prediction_matrix = self.prediction_matrix(inputs)
        new_matrix = self.concatenate_matrix(prediction_matrix,inputs)
        self.meta_Alg.train(new_matrix,outputs)

    ##  train each component model on the input data  ##
    def fold_train(self, x_train, y_train):
        for i in range(len(self.model_list)):
            self.model_list[i].train(x_train,y_train)

    ##  gets prediction from each componenet model, and feeds these  ##
    ##  into meta algorithm for overall prediction  ##
    def predict(self,x_test):
        pred_matrix = self.prediction_matrix(x_test)
        new_matrix = self.concatenate_matrix(pred_matrix,x_test)
        return self.meta_Alg.predict(new_matrix) 

    ##  gets matrix of predicitions for each input  ##
    def prediction_matrix(self, x_test):
        prediction_matrix = []
        for j in range(len(x_test)):
            models_prediction = []
            for i in range(len(self.model_list)):
                curr_sample = np.array([x_test[j]])
                models_prediction.append(self.model_list[i].predict(curr_sample)[0])
            prediction_matrix.append(models_prediction)
        return np.asarray(prediction_matrix)

    ##  gets accuracy for each model in input ##
    def fold_accuracy(self, x_test, y_test):
        accuracies = []
        for i in range(len(self.model_list)):
            prediction = self.model_list[i].report_accuracy(x_test, y_test)
            accuracies.append(prediction)
        return accuracies

    ##  returns accuracy from best model  ##
    def report_accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        results = accuracy_score(predictions, y_test)
        return results
