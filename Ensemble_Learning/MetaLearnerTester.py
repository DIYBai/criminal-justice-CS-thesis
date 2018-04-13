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

    #this method run the dataset on meta learner multiple iterations
    def run_trials(self, filename, model,iterations, inputs, outputs):
        test_accuracies=[]
        train_accuracies=[]
        fp_rate_test=[]
        fp_rate_train=[]
        fn_rate_test=[]
        fn_rate_train=[]
        for i in range(0, iterations):
            #reset the meta learner's attributes for each iteration
            self.split_data(inputs,outputs)
            model.models=[]
            model.settings=[]
            model.meta_input=[]
            model.meta_output=[]
            model.x_test=self.x_test
            model.y_test=self.y_test
            model.meta_results=[]
            model.train(self.x_train, self.y_train)
            test_accuracies.append(model.report_accuracy(self.x_test,self.y_test))
            fp_rate_test.append(model.false_positive_rate)
            fn_rate_test.append(model.false_negative_rate)
            train_accuracies.append(model.report_accuracy(self.x_train,self.y_train))
            fp_rate_train.append(model.false_positive_rate)
            fn_rate_train.append(model.false_negative_rate)
        self.calculate(filename,model,test_accuracies, train_accuracies, fp_rate_test,fp_rate_train,fn_rate_test,fn_rate_train)
        return

    #this function export results to files
    def export_results(self, filename, test_metrics, train_metrics):
        header=["Model","File_name","Test_set","Average","Std_dev","Fp_rate","Fn_rate"]
        items=[test_metrics,train_metrics]
        dataframe=pd.DataFrame(items,columns=header)
        dataframe.to_csv("mlTester_Metrics/broward_with_race_forposter_Meta_result_ANNs.csv",mode='a')
        return

    #this function does mathematical calculation such as false_positive_rate, standard deviation, etc.
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
        #get the results to export_results to save to file
        return self.export_results(filename, test_metrics, train_metrics)

    #this function split the dataset
    def split_data(self, inputs, outputs):
        #needs a component model object declaration to use split_data()
        test_DTree = Dtree()
        self.x_train, self.x_test, self.y_train, self.y_test = test_DTree.split_data(inputs, outputs, .5)
        print len(self.x_test), " Y TEST LENGTH"
