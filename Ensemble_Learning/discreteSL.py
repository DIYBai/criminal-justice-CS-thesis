import numpy as np
import MetaModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import copy

class discreteSL(MetaModel.MetaModel):


    def __init__(self, model_list):
        self.model_list = model_list
        self.model_accuracies = []
        self.best_model = model_list[0]

    def concatenate(self, folds, i):
        folds_copy = copy.deepcopy(folds)
        test = folds_copy.pop(i)
        return np.concatenate(folds_copy, axis=0), test

    def mean_accuracies(self,accuracies_list,b):
        self.model_accuracies = [sum(x)/b for x in zip(*accuracies_list)]

    def select_model(self):
        high_acc = 0
        high_indx = 0
        for i in range(len(self.model_list)):
            current_acc = self.model_accuracies[i]
            if(high_acc < current_acc):
                high_acc = current_acc
                high_indx = i
        self.best_model = self.model_list[i]


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
        self.select_model()
        print(self.model_accuracies)
        print(self.best_model)      

    def fold_train(self, x_train, y_train):
        for i in range(len(self.model_list)):
            self.model_list[i].train(x_train,y_train)

    def get_prediction(self, x_test):
        predictions = []
        for i in range(len(self.model_list)):
            prediction = self.model_list[i].predict(x_test)
            predictions.append(prediction)

        predictions_np = np.asarray(predictions)
        m = stats.mode(predictions_np)
        return m.mode[0]

    def fold_accuracy(self, x_test, y_test):
        accuracies = []
        for i in range(len(self.model_list)):
            prediction = self.model_list[i].report_accuracy(x_test, y_test)
            accuracies.append(prediction)
        return accuracies

    def report_accuracy(self, x_test, y_test):
        return self.best_model.report_accuracy(x_test, y_test)

