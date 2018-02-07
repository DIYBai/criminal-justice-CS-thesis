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

    def train(self, x_train, y_train):
        self.Dtree(x_train,y_train)
        #self.ANN(x_train,y_train)
        #self.KNN(x_train,y_train)
        #self.LogR(x_train,y_train)
        return
    def predict(self, x_test):
        predictions = []
        predictions_number =[]
        for model in self.models:
            prediction = model.predict(x_test)
            predictions.append([model,prediction[0]])
            predictions_number.append(prediction[0])

        predictions_np = np.asarray(predictions)
        predictions_number_np = np.asarray(predictions_number)
        m = stats.mode(predictions_number_np)
        print predictions_number, "predictions from list of models"
        print m[0]
        #return m.mode[0]
        #apocalypticDoc
        return predictions_np
    def report_accuracy(self, x_test, y_test):
        for i in range(0, len(y_test)):
            prediction = self.predict([x_test[i]])
            #print prediction
        return
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
                for i in range(5,31,5):
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
