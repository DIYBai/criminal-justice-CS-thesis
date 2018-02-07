import numpy as np
import EnsembleModel
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR

class Meta_Learner(EnsembleModel.EnsembleModel):
    def __init__(self):
        self.models=[]
        self.settings=[]
        self.meta_input=[]
        self.meta_output=[]

    def train(self, x_train, y_train):
        self.Dtree(x_train,y_train)
        #self.ANN(x_train,y_train)
        #self.KNN(x_train,y_train)
        #self.LogR(x_train,y_train)
        self.train_meta(x_train,y_train)
        #my_ANN = ANN(layers=[100,90,80])
        #x_train, x_test, y_train, y_test = my_ANN.split_data(self.meta_input, self.meta_output, .25)
        #my_ANN.train(x_train,y_train)
        return

    def predict(self,x_test):
        self.meta_input=[]
        #self.meta_output=[]
        #predictions=[]
        #for model in self.models:
        #    prediction = model.predict(x_test)
        #    predictions.append(prediction[0])
        #self.meta_input.append(predictions)
        self.build_metadata(x_test)
        return self.my_ANN.predict(self.meta_input)



    def report_accuracy(self, x_test, y_test):
        print "In report accuracy"
        count = 0
        f_neg =0
        f_pos =0
        print "Length", len(y_test)
        for i in range(0, len(y_test)):
            prediction = self.predict([x_test[i]])
            if prediction == y_test[i]:
               count += 1
            #1 for yes ,0for no
            #false negative
            elif prediction == 0.0 and y_test[i] == 1.0:
                f_neg +=1
            elif prediction == 1.0 and y_test[i] == 0.0:
                f_pos +=1
        self.error_type(float(f_neg),float(f_pos),y_test)

        return float(count)/float(len(y_test))

    def error_type(self,f_neg,f_pos,y_test):
        tn=0.0
        tp=0.0
        m = stats.mode(np.asarray(y_test))
        if m[0]==0.0:
            print m
            tn=float(m[1])
            tp=float(len(y_test)-m[1])
        else:
            print m
            tp=float(m[1])
            tn=float(len(y_test)-m[1])
        false_positive=(f_neg)/(f_neg+tn)
        false_negative=(f_pos)/(f_pos+tp)
        print "False Positive Rate: ", false_positive, "False Negative Rate: ",false_negative

    def build_metadata(self, x_test):
        predictions = []
        predictions_number =[]
        for model in self.models:
            prediction = model.predict(x_test)
            predictions.append([model,prediction[0]])
            predictions_number.append(prediction[0])

        predictions_np = np.asarray(predictions)
        predictions_number_np = np.asarray(predictions_number)
        m = stats.mode(predictions_number_np)
        #print predictions_number, "predictions from list of models"
        #the resulting predictions from each model will be used as inputs
        self.meta_input.append(predictions_number)
        #the mode of the resulting predictions from each model will be used as outputs
        #self.meta_output.append(m[0])
        #print m[0]
        #return m.mode[0]
        #apocalypticDoc
        return predictions_np

    def train_meta(self, x_test,y_test):
        print "In Train Meta"
        for i in range(0, len(y_test)):
            prediction = self.build_metadata([x_test[i]])
        self.my_ANN = ANN(layers=(100,))
        x_train, x_test, y_train, y_test = self.my_ANN.split_data(self.meta_input, y_test, .25)
        self.my_ANN.train(x_train,y_train)
        print "FINAL META ACCURACY",self.my_ANN.report_accuracy(x_test,y_test)

    def report_individual_accuracy(self, x_test, y_test):
        results=[]
        for model in self.models:
            results.append([model,model.report_accuracy(x_test,y_test)])
        #print self.settings
        return results
    def ANN(self,x_train,y_train):
        algorithms = ['lbfgs', 'sgd', 'adam']
        activation_f = ['identity', 'logistic', 'tanh', 'relu']
        learning_method = ['constant', 'invscaling', 'adaptive']
        alpha= [.0001, .0005, .0009 , .001 , .005, .009 , .01]
        rate =[.001 , .005 , .009 , .01 , .05 ,.09 , .1 , .0009 , .0005 , .0001]
        layers = [(100,),(100, 90,),(100,90,80,)]
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
                                    #current_model.train(self.x_train,self.y_train)
                                    self.models.append(current_model)
                                    current_model.train(x_train,y_train)
                                    self.settings.append(["ANN",algo, function,methods, i , r, a,layer])
                                    #print("Current Settings: ",algo, function,methods, i , r, a,layer)
                                    #print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                                    #print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
    def Dtree(self, x_train, y_train):
        criteria =['gini','entropy']
        split = ['best', 'random']
        depth = 30
        for crit in criteria:
            for strategy in split:
                for i in range(5,16,5):
                    current_model=Dtree(crit, strategy , i)
                    #new stuff
                    #self.current_model=current_model
                    self.models.append(current_model)
                    current_model.train(x_train,y_train)
                    self.settings.append(["Dtree",crit, strategy , i])
                    #print("Current Settings: ",crit, strategy , i)
                    #print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                    #print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
    def KNN(self, x_train,y_train):
        weights =['uniform','distance']
        algorithms = ['auto','ball_tree','kd_tree','brute']
        for i in range(2,20,2):
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
    def LogR(self, x_train,y_train):
        algorithms=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        for algo in algorithms:
            for i in range(1,502, 100):
                current_model=LogR(algo,i)
                self.models.append(current_model)
                current_model.train(x_train,y_train)
                self.settings.append(["LogR",algo,i])
                #current_model.train(self.x_train,self.y_train)
                #print("Current Settings: ",algo,i)
                #print("Naive Ensemble Test accuracy: test data set",current_model.report_accuracy(self.x_test,self.y_test))
                #print("Naive Ensemble Test accuracy: train data set",current_model.report_accuracy(self.x_train,self.y_train))
