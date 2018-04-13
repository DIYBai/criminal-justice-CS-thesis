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
        self.meta_results=[]

    #this method trains the whole model, including component and meta learner
    #takes in training set as input and does not return anything
    def train(self, x_train, y_train):
        self.x_train=x_train
        self.y_train=y_train
        self.train_best_KNNs(x_train,y_train)
        self.train_best_LogRs(x_train,y_train)
        self.train_best_Dtrees(x_train, y_train)
        self.train_best_ANNs(x_train,y_train)
        self.train_meta(x_train,y_train)
        return

    #this method makes predictions for the meta learner
    #takes in inputs from testing set and return
    def predict(self,x_test):
        self.meta_input=[]
        #calls build_metadata() to get the outputs from the component models and
        #then combine the outputs to x_test
        self.build_metadata(x_test)
        return self.my_ANN.predict(self.meta_input)

    #this method report the accuracy for the meta learner, takes in testing dataset
    #takes in the testing subset and return a floating point value representign accuracy
    def report_accuracy(self, x_test, y_test):
        count = 0
        f_neg =0
        f_pos =0
        results=self.report_individual_accuracy(x_test,y_test)
        train_results=self.report_individual_accuracy(self.x_train,self.y_train)
        inputs=self.combine_aray(train_results,results)
        inputs2=self.combine_aray(self.settings,inputs)
        inputs_np=np.array(inputs2)
        self.save_results(inputs, self.filename)
        #this loop keeps track the count of correct guesses and the count of false negatives
        #and false positives
        for i in range(0, len(y_test)):
            prediction = self.predict([x_test[i]])
            if prediction == y_test[i]:
               count += 1
            #false negative, if actual is positive and classifier guessed negative
            elif prediction == 0.0 and y_test[i] == 1.0:
                f_neg +=1
            #false positive, if actual is negative and classifier guessed positive
            elif prediction == 1.0 and y_test[i] == 0.0:
                f_pos +=1
        self.error_type(float(f_neg),float(f_pos),y_test)
        return float(count)/float(len(y_test))

    #this function combiens two array
    def combine_aray(self, list1,list2):
        for i in range(0,len(list1)):
            for element in list1[i]:
                list2[i].append(element)
        return list2


    #this function calculates the false_negative_rate and false_positive_rate
    #takes in the label of test subset and the count of f_neg and f_pos
    def error_type(self,f_neg,f_pos,y_test):
        tn=0.0
        tp=0.0
        m = stats.mode(np.asarray(y_test))
        #tn is all the negative minus the false positive
        tn=float(m[1])-f_pos
        #tp is all the positive minus the false negative
        tp=(float(len(y_test)-m[1]))-f_neg
        if (f_pos+tn)==0 and (f_neg+tp)==0:
            self.false_positive_rate=0.0
            self.false_negative_rate=0.0
        elif (f_pos+tn)==0:
            self.false_positive_rate=0.0
            self.false_negative_rate=(f_neg)/(f_neg+tp)
        elif (f_neg+tp)==0:
            self.false_negative_rate=0.0
            self.false_positive_rate=(f_pos)/(f_pos+tn)
        else:
            self.false_positive_rate=(f_pos)/(f_pos+tn)
            self.false_negative_rate=(f_neg)/(f_neg+tp)
        print "False Positive Rate: ", self.false_positive_rate, "False Negative Rate: ",self.false_negative_rate

    #this function build the input dataset for the metalearner by combiningthe metadata to the inputs
    def build_metadata(self, x_test):
        predictions = []
        predictions_number =[]
        #x_test[0] contains a single row of input from test data set
        for element in x_test[0]:
            predictions_number.append(element)
        #loop through all the component models
        for model in self.models:
            #prediction=single prediction
            prediction = model.predict(x_test)
            #predictions array has the correspoinding model and its prediction
            predictions.append([model,prediction[0]])
            #prediction number has the row of input plus the predictions from each model
            predictions_number.append(prediction[0])
        #some dataprocessing and conversions to numpy arrays
        predictions_np = np.asarray(predictions)
        self.meta_results.append(predictions_np)
        self.save_results(predictions_np, self.filename)
        predictions_number_np = np.asarray(predictions_number)
        m = stats.mode(predictions_number_np)
        #self.meta_input will contains all inputs from the dataset plus the predictions of the
        #component models
        self.meta_input.append(predictions_number_np)

    #this method trains the actual meta learner
    def train_meta(self, x_test,y_test):
        #loop through all the rows in the dataset and get the prediction from component model_selection
        #then append to it and add each row to self.meta_input
        for i in range(0, len(self.y_test)):
            prediction = self.build_metadata([self.x_test[i]])
        self.save_results(self.meta_results, "New_data_predictions-LogR-only.csv")
        self.my_ANN = ANN('adam', 'logistic','constant', 700 , .01, .0001,(100,90,80,70))
        #finally the dataset with the meta data gets feed into an ANN
        x_train, x_test, y_train, y_test = self.my_ANN.split_data(self.meta_input, self.y_test, .25)
        self.my_ANN.train(x_train,y_train)
        print "FINAL META ACCURACY",self.my_ANN.report_accuracy(x_test,y_test)

    #this function simply report the accuracy for the component models
    def report_individual_accuracy(self, x_test, y_test):
        results=[]
        for model in self.models:
            results.append([model,model.report_accuracy(x_test,y_test)])
        return results

    #this function saves the data to a file
    def save_results(self, items,filename):
        dataframe=pd.DataFrame(items)
        dataframe.to_csv(filename, mode= 'a')

    #this function trains the component ANN models with the best settings
    def train_best_ANNs(self, x_train,y_train):
        ANNs=[ANN('adam', 'logistic','constant', 700 , .01, .0001,(100,90,80,70)),ANN('adam', 'tanh','constant', 6700 , .001, .0001,(100,90,80,70)),
        ANN('adam', 'tanh','adaptive', 8700 , .0001, .0009,(100,90,80,70))]
        for model in ANNs:
            self.models.append(model)
            model.train(x_train,y_train)
            self.settings.append(["ANN"])

    #this function trains the component Decesion Tree models with the best settings
    def train_best_Dtrees(self, x_train, y_train):
        Dtrees=[Dtree('gini', 'best' , 3),Dtree('entropy', 'best' , 3),Dtree('entropy', 'random' , 3)]
        for tree in Dtrees:
            self.models.append(tree)
            tree.train(x_train,y_train)
            self.settings.append(["Dtree"])

    #this function trains the component Decesion Tree models with the best settings
    def train_best_KNNs(self, x_train,y_train):
        KNNs= [KNN(10,'uniform', 'auto'),KNN(15,'uniform', 'kd_tree'),KNN(20,'uniform', 'ball_tree')]
        for knn in KNNs:
            self.models.append(knn)
            knn.train(x_train,y_train)
            self.settings.append(["KNN"])

    #this function trains the component Decesion Tree models with the best settings
    def train_best_LogRs(self, x_train,y_train):
        LogRs=[LogR('newton-cg',3),LogR('newton-cg',6),LogR('newton-cg',9)]
        for log in LogRs:
            self.models.append(log)
            log.train(x_train,y_train)
            self.settings.append(["LogR"])
