import EnsembleModel
import os
import sys
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats
lib_path = os.path.abspath(os.path.join('..','ComponentModels'))
sys.path.append(lib_path)
from ANN import ANN
import Tester
import copy
import numpy as np


class slTester(Tester.Tester):

##handel convergence warning 
##too string method in each component model

    def test_sample(self,model,inputs,outputs):
        model.train(inputs, outputs)
        predictions = model.predict(inputs)
        tn, fp, fn, tp = confusion_matrix(predictions,outputs).ravel()
        a_score = accuracy_score(predictions,outputs)
        return a_score,[tn,fp,fn,tp]

    def update_metrics(self, cm, conf, acc, a):
    	##print(acc)
        update_cm = [x + y for x, y in zip(conf, cm)]
        acc.append(a)
        return update_cm, acc


    def get_result_metrics(self, accuracies, cm, iterations):

        final_cm = [x / iterations for x in cm]
    	std = np.std(accuracies)
    	fp_rate = final_cm[1]/(final_cm[1]+final_cm[2])
    	fn_rate = final_cm[2]/(final_cm[1]+final_cm[2])
    	acc = (np.sum(accuracies))/iterations

    	return [str(acc),str(std),str(fp_rate),str(fn_rate)]


    def run_trials(self,filename, model, iterations, inputs, outputs):
        parser = ANN()
        train_acc = [0]
        test_acc = [0]
        train_cm = [0,0,0,0]
        test_cm = [0,0,0,0]
        for i in range(iterations):
            x_train, x_test, y_train, y_test = parser.split_data(inputs, outputs, .25)
            train_a, train_conf = self.test_sample(model, x_train, y_train)
            test_a, test_conf = self.test_sample(model, x_test, y_test)
            train_cm, train_acc = self.update_metrics(train_cm, train_conf, train_acc, train_a)
            test_cm, test_acc = self.update_metrics(test_cm, test_conf, test_acc, test_a)
        
        test_metrics = self.get_result_metrics(test_acc,test_cm,iterations)
        train_metrics = self.get_result_metrics(train_acc,train_cm,iterations)
        self.export_results(filename, test_metrics, train_metrics)
        
        return 

    ##, test_metrics, train_metrics
    def export_results(self,filename, test_metrics, train_metrics):
        ##print(test_metrics)
        ##print(train_metrics)
        with open(r'Ensemble_Metrics.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(test_metrics)
        return

