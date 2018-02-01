import numpy as np
import MetaModel
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

class NaiveEnsemble(MetaModel.MetaModel):


    def __init__(self, model_list):
        self.model_list = model_list

    def train(self, x_train, y_train):
        for i in range(len(self.model_list)):
            print self.model_list[i], " Model"
            self.model_list[i].train(x_train,y_train)

    def get_prediction(self, x_test):
        #predictions = np.zeros(len(self.model_list),)
        predictions=[]
        for i in range(len(self.model_list)):
            prediction = self.model_list[i].predict(x_test)
            #np.append(predictions,prediction)
            predictions.append(prediction)
        #if predictions[0]==1.0 or predictions[1]==1.0 or predictions[2]==1.0 or predictions[3]==1.0:
        #if predictions[0]==0.0 and predictions[1]==0.0 and predictions[2]==0.0 and predictions[3]==0.0:
            #print "Prediction matrix: ", predictions
        m = stats.mode(predictions)
        #return m.mode[0]
        #if m[0][0]!=0.0:
        print m[0][0]
        return m[0][0]

    def report_accuracy(self, x_test, y_test):
        count = 0
        for i in range(0, len(y_test)):
            #print [x_test[i]], " PAssing in Values"
            prediction = self.get_prediction(([x_test[i]]))
            #print prediction," Iteration: " , i
            if prediction == y_test[i]:
               count += 1


        return float(count)/float(len(y_test))

    def report_individual_accuracy(self,x_test,y_test):
        temp=[]
        for i in range(0,len(self.model_list)):
            acc=self.model_list[i].report_accuracy(x_test,y_test)
            temp.append(acc)
        return temp
