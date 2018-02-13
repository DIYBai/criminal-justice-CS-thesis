import numpy as np
import EnsembleModel
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

class NaiveEnsemble(EnsembleModel.EnsembleModel):


    def __init__(self, model_list):
        self.model_list = model_list

    def train(self, x_train, y_train):
        for i in range(len(self.model_list)):
            # print("Training", self.model_list[i])
            self.model_list[i].train(x_train,y_train)

    def predict(self, x_test):
        predictions = []
        for i in range(len(self.model_list)):
            prediction = self.model_list[i].predict(x_test)
            # print(prediction)
            predictions.append(prediction)

        predictions_np = np.asarray(predictions)
        m = stats.mode(predictions_np)
        return m.mode[0]

    #similar functionality to predict BUT returns the prediction matrix, not just the actual prediction
    #this is so that other code can access the component models' predictions
    #to get the actual prediction, use (np.average(FUNCTIONOUTPUT, axis = 0))[0][0]
    def predict_prob(self, x_test):
        probabilities = []
        for i in range(len(self.model_list)):
            prob = self.model_list[i].predict_prob(x_test)
            # print(self.model_list[i], '\t', '{:.3f}'.format(prob[0][0]) ) #temporary testing
            probabilities.append(prob)

        probs_np = np.asarray(probabilities)
        # return np.average(probs_np, axis = 0)
        return probs_np

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
