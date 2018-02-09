import numpy as np
import EnsembleModel
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

class NaiveEnsemble(EnsembleModel.EnsembleModel):


    def __init__(self, model_list):
        self.model_list = model_list

    def train(self, x_train, y_train):
        for i in range(len(self.model_list)):
            self.model_list[i].train(x_train,y_train)

    def predict(self, x_test):
        predictions = []
        for i in range(len(self.model_list)):
            prediction = self.model_list[i].predict(x_test)
            predictions.append(prediction)

        predictions_np = np.asarray(predictions)
        m = stats.mode(predictions_np)
        return m.mode[0]

    def report_accuracy(self, x_test, y_test):
        count = 0
        for i in range(0, len(y_test)):
            prediction = self.predict([x_test[i]])
            if prediction == y_test[i]:
               count += 1

        return float(count)/float(len(y_test))

    def report_individual_accuracy(self,x_test,y_test):
        temp=[]
        for i in range(0,len(self.model_list)):
            acc=self.model_list[i].report_accuracy(x_test,y_test)
            temp.append(acc)
        return temp
