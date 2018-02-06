import numpy as np
import MetaModel
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR

class Meta_Learner:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
    def ANN(self):
        algorithms = ['lbfgs', 'sgd', 'adam']
        activation_f = ['identity', 'logistic', 'tanh', 'relu']
        learning_method = ['constant', 'invscaling', 'adaptive']
        alpha= [.0001, .0005, .0009 , .001 , .005, .009 , .01]
        rate =[.001 , .005 , .009 , .01 , .05 ,.09 , .1 , .0009 , .0005 , .0001]
        layers = [[100],[100, 90],[100,90,80]]
        for algo in algorithms:
            for function in activation_f:
                for methods in learning_method:
                    #iterations loop
                    for i in range(200,301,100):
                        for r in rate:
                            for a in alpha:
                                #layers
                                for layer in layers:
                                    current_model=ANN(algo, function,methods, i , r, a,layer)
                                    current_model.train(self.x_train,self.y_train)
                                    print("Current Settings: ",algo, function,methods, i , r, a,layer)
                                    print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                                    print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
    def Dtree(self):
        criteria =['gini','entropy']
        split = ['best', 'random']
        depth = 30
        for crit in criteria:
            for strategy in split:
                for i in range(5,31,5):
                    current_model=Dtree(crit, strategy , i)
                    current_model.train(self.x_train,self.y_train)
                    print("Current Settings: ",crit, strategy , i)
                    print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                    print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
    def KNN(self):
        weights =['uniform','distance']
        algorithms = ['auto','ball_tree','kd_tree','brute']
        for i in range(2,20,2):
            for w in weights:
                for algo in algorithms:
                    current_model=KNN(i ,w, algo)
                    current_model.train(self.x_train,self.y_train)
                    print("Current Settings: ",i ,w, algo)
                    print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                    print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
    def LogR(self):
        algorithms=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        for algo in algorithms:
            for i in range(1,502, 100):
                current_model=LogR(algo,i)
                current_model.train(self.x_train,self.y_train)
                print("Current Settings: ",algo,i)
                print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
