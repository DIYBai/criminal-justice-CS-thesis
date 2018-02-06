import numpy as np
import MetaModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import copy

class fullSL(EnsembleModel.EnsembleModel):


    def __init__(self, model_list):
        self.model_list = model_list
        self.model_accuracies = []
        self.best_model = model_list[0]
        self.total_accuracy = 100.000

    def concatenate(self, folds, i):
        folds_copy = copy.deepcopy(folds)
        test = folds_copy.pop(i)
        return np.concatenate(folds_copy, axis=0), test

    def mean_accuracies(self,accuracies_list,b):
        self.model_accuracies = [sum(x)/b for x in zip(*accuracies_list)]
        accuracies_list_np = np.asarray(accuracies_list)
        self.total_accuracy = np.sum(accuracies_list_np)


    def train(self, inputs, outputs, b=10):
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
        print(self.model_accuracies)
        print(self.best_model)

    def fold_train(self, x_train, y_train):
        for i in range(len(self.model_list)):
            self.model_list[i].train(x_train,y_train)

    def predict(self, x_test):
        total_accuracy = 0.0
        for i in range(len(self.model_list)):
            curr_accuracy = self.model_list[i].predict(x_test)
            print(curr_accuracy)
            total_accuracy += (curr_accuracy * self.model_accuracies[i])
        if total_accuracy >= (self.total_accuracy/2.0): return 1
        return 0

    def fold_accuracy(self, x_test, y_test):
        accuracies = []
        for i in range(len(self.model_list)):
            prediction = self.model_list[i].report_accuracy(x_test, y_test)
            accuracies.append(prediction)
        return accuracies

    def report_accuracy(self, x_test, y_test):
        count = 0
        for i in range(0, len(y_test)):
            prediction = self.predict([x_test[i]])
            if prediction == y_test[i]:
               count += 1

        return float(count)/float(len(y_test))
