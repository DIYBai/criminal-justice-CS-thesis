import numpy as np
import pandas as pd
import Tester
import os
import sys
lib_path = os.path.abspath( os.path.join('..', 'ComponentModels') )
sys.path.append(lib_path)
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR

class Meta_Learner_Tester(Tester.Tester):

    def run_trials(self, filename, model,iterations, inputs, outputs):
        test_accuracies=[]
        train_accuracies=[]
        fp_rate_test=[]
        fp_rate_train=[]
        fn_rate_test=[]
        fn_rate_train=[]
        for i in range(0, iterations):
            self.split_data(inputs,outputs)
            model.models=[]
            model.settings=[]
            model.meta_input=[]
            model.meta_output=[]
            model.filename=filename
            model.meta_results=[]
            model.train(self.x_train, self.y_train)
            test_accuracies.append(model.report_accuracy(self.x_test,self.y_test))
            fp_rate_test.append(model.false_positive_rate)
            fn_rate_test.append(model.false_negative_rate)
            train_accuracies.append(model.report_accuracy(self.x_train,self.y_train))
            fp_rate_train.append(model.false_positive_rate)
            fn_rate_train.append(model.false_negative_rate)
        self.calculate(filename,model,test_accuracies, train_accuracies, fp_rate_test,fp_rate_train,fn_rate_test,fn_rate_train)
        print "Finish run trial"
        return

    def export_results(self, filename, test_metrics, train_metrics):
        header=["Model","File_name","Test_set","Average","Std_dev","Fp_rate","Fn_rate"]
        items=[test_metrics,train_metrics]
        dataframe=pd.DataFrame(items,columns=header)
        dataframe.to_csv("Long/MetaLearner_Result.csv")
        return

    def calculate(self,filename,model,test_accuracies, train_accuracies, fp_rate_test,fp_rate_train,fn_rate_test,fn_rate_train):
        test_std = np.std(np.asarray(test_accuracies))
        train_std = np.std(np.asarray(train_accuracies))
        test_average = np.average(np.asarray(test_accuracies))
        train_average = np.average(np.asarray(train_accuracies))
        fp_test = np.average(np.asarray(fp_rate_test))
        fp_train = np.average(np.asarray(fp_rate_train))
        fn_test = np.average(np.asarray(fn_rate_test))
        fn_train = np.average(np.asarray(fn_rate_train))
        test_metrics=[model, filename, "Testing", test_average, test_std, fp_test,fn_test]
        train_metrics=[model, filename, "Training", train_average, train_std, fp_train,fn_train]
        return self.export_results(filename, test_metrics, train_metrics)

    def split_data(self, inputs, outputs):
        test_DTree = Dtree()
        self.x_train, self.x_test, self.y_train, self.y_test = test_DTree.split_data(inputs, outputs, .25)
