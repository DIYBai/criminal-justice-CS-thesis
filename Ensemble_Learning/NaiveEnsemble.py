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
        predictions = np.zeros(len(self.model_list),)
        for i in range(len(self.model_list)):
            prediction = self.model_list[i].predict(x_test)
            np.append(predictions,prediction)

        m = stats.mode(predictions)
        #return m.mode[0]
        return m[0][0]

    def report_accuracy(self, x_test, y_test):
        count = 0
        for i in range(0, len(y_test)):
            prediction = self.get_prediction(x_test)
            if prediction == y_test[i]:
               count += 1

        return float(count)/float(len(y_test))

    def report_individual_accuracy(self,x_test,y_test):
        temp=[]
        for i in range(len(self.model_list)):
            acc=self.model_list[i].report_accuracy(x_test,y_test)
            temp.append(acc)
        return temp
