import numpy as np
import pandas as pd
import EnsembleModel
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

class Meta_Learner(EnsembleModel.EnsembleModel):
    def __init__(self,f_name):
        self.models=[]
        self.settings=[]
        self.meta_input=[]
        self.meta_output=[]
        self.filename=f_name
        #self.x_train=x_t
        #self.y_train=y_t
        self.meta_results=[]

    def train(self, x_train, y_train):
        self.x_train=x_train
        self.y_train=y_train
        #self.train_ANNs(x_train,y_train)
        #print "finish ANN"
        #self.train_Dtrees(x_train,y_train)
        #self.train_KNNs(x_train,y_train)
        self.train_LogRs(x_train,y_train)
        self.train_meta(x_train,y_train)
        return

    def predict(self,x_test):
        self.meta_input=[]
        self.build_metadata(x_test)
        return self.my_ANN.predict(self.meta_input)



    def report_accuracy(self, x_test, y_test):
        print "In report accuracy"
        count = 0
        f_neg =0
        f_pos =0
        results=self.report_individual_accuracy(x_test,y_test)
        train_results=self.report_individual_accuracy(self.x_train,self.y_train)
        inputs=self.combine_aray(train_results,results)
        inputs2=self.combine_aray(self.settings,inputs)
        inputs_np=np.array(inputs2)
        self.save_results(inputs, self.filename)
        print "Length", len(y_test)
        for i in range(0, len(y_test)):
            prediction = self.predict([x_test[i]])
            if prediction == y_test[i]:
               count += 1
            #1 for yes ,0for no
            #false negative, if actual is positive and classifier guessed negative
            elif prediction == 0.0 and y_test[i] == 1.0:
                f_neg +=1
            #false positive, if actual is negative and classifier guessed positive
            elif prediction == 1.0 and y_test[i] == 0.0:
                f_pos +=1
        self.error_type(float(f_neg),float(f_pos),y_test)
        return float(count)/float(len(y_test))

    def combine_aray(self, list1,list2):
        for i in range(0,len(list1)):
            for element in list1[i]:
                list2[i].append(element)
        return list2



    def error_type(self,f_neg,f_pos,y_test):
        tn=0.0
        tp=0.0
        m = stats.mode(np.asarray(y_test))
        #if false
        if m[0]==0.0:
            print m
            #tn is all the negative minus the false positive
            tn=float(m[1])-f_pos
            #tp is all the positive minus the false negative
            tp=(float(len(y_test)-m[1]))-f_neg
        #if true
        #else:
            #print m
            #tp=float(m[1])
            #tn=float(len(y_test)-m[1])
        self.false_positive_rate=(f_pos)/(f_pos+tn)
        self.false_negative_rate=(f_neg)/(f_neg+tp)
        print "False Positive Rate: ", self.false_positive_rate, "False Negative Rate: ",self.false_negative_rate

    def build_metadata(self, x_test):
        predictions = []
        predictions_number =[]
        for element in x_test[0]:
            predictions_number.append(element)
        for model in self.models:
            prediction = model.predict(x_test)
            predictions.append([model,prediction[0]])
            predictions_number.append(prediction[0])
        predictions_np = np.asarray(predictions)
        self.meta_results.append(predictions_np)
        #self.save_results(predictions_np)
        predictions_number_np = np.asarray(predictions_number)
        m = stats.mode(predictions_number_np)
        #print predictions_number_np
        self.meta_input.append(predictions_number_np)
        #return predictions_np

    def train_meta(self, x_test,y_test):
        print "In Train Meta"
        for i in range(0, len(y_test)):
            prediction = self.build_metadata([x_test[i]])
        self.save_results(self.meta_results, "New_data_predictions-LogR-only.csv")
        self.my_ANN = ANN(layers=(100,90,80,70),learning_method='adaptive',iteration=1000)
        #self.my_ANN = Dtree('gini','best',10)
        x_train, x_test, y_train, y_test = self.my_ANN.split_data(self.meta_input, y_test, .25)
        self.my_ANN.train(x_train,y_train)
        print "FINAL META ACCURACY",self.my_ANN.report_accuracy(x_test,y_test)

    def report_individual_accuracy(self, x_test, y_test):
        results=[]
        for model in self.models:
            results.append([model,model.report_accuracy(x_test,y_test)])
        return results

    def save_results(self, items,filename):
        dataframe=pd.DataFrame(items)
        dataframe.to_csv(filename)
    def train_ANNs(self,x_train,y_train):
        #algorithms = ['lbfgs', 'sgd', 'adam']
        algorithms = [ 'adam']
        #activation_f = ['identity', 'logistic', 'tanh', 'relu']
        activation_f = ['relu']
        learning_method = ['constant', 'invscaling', 'adaptive']
        #learning_method = ['constant','adaptive']
        #alpha= [.0001, .0005, .0009 , .001 , .005, .009 , .01]
        alpha= [.0001, .0009,.00001]
        #rate =[.001 , .005 , .009 , .01 , .05 ,.09 , .1 , .0009 , .0005 , .0001]
        rate =[.001, .01,.0001]
        #layers = [(100,),(100, 90,),(100,90,80,),(100,90,80,70,)]
        layers = [(100, 90,),(100,90,80,70,)]
        for algo in algorithms:
            for function in activation_f:
                for methods in learning_method:
                    #iterations loop
                    for i in range(700,10000,1000):
                        for r in rate:
                            for a in alpha:
                                #layers
                                for layer in layers:
                                    current_model=ANN(algo, function,methods, i , r, a,layer)
                                    #current_model.train(self.x_train,self.y_train)
                                    self.models.append(current_model)
                                    current_model.train(x_train,y_train)
                                    self.settings.append(["ANN",algo, function,methods, i , r, a,layer])
                                    #print("Current Settings: ",algo, function,methods, i , r, a,layer)
                                    #print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                                    #print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
    def train_Dtrees(self, x_train, y_train):
        criteria =['gini','entropy']
        split = ['best', 'random']
        depth = 30
        for crit in criteria:
            for strategy in split:
                for i in range(5,41,5):
                    current_model=Dtree(crit, strategy , i)
                    #new stuff
                    #self.current_model=current_model
                    self.models.append(current_model)
                    current_model.train(x_train,y_train)
                    self.settings.append(["Dtree",crit, strategy , i])
                    #print("Current Settings: ",crit, strategy , i)
                    #print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                    #print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
    def train_KNNs(self, x_train,y_train):
        weights =['uniform','distance']
        algorithms = ['auto','ball_tree','kd_tree','brute']
        for i in range(5,21,5):
            for w in weights:
                for algo in algorithms:
                    current_model=KNN(i ,w, algo)
                    self.models.append(current_model)
                    current_model.train(x_train,y_train)
                    self.settings.append(["KNN",i ,w, algo])
                    #current_model.train(self.x_train,self.y_train)
                    #print("Current Settings: ",i ,w, algo)
                    #print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                    #print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
    def train_LogRs(self, x_train,y_train):
        #algorithms=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        algorithms=['newton-cg']
        for algo in algorithms:
            for i in range(1,202, 100):
                current_model=LogR(algo,i)
                self.models.append(current_model)
                current_model.train(x_train,y_train)
                self.settings.append(["LogR",algo,i])
                #current_model.train(self.x_train,self.y_train)
                #print("Current Settings: ",algo,i)
                #print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                #print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
