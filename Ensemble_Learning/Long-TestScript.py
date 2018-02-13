import RiskParser as rp
import os
import sys
lib_path = os.path.abspath( os.path.join('..', 'ComponentModels') )
sys.path.append(lib_path)
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR
from NaiveEnsemble import NaiveEnsemble
import numpy as np
from Meta_Learner import Meta_Learner



inputs, outputs = rp.parse_data("../Data/RiskAssessData.csv")
print(inputs)
print(outputs)
print("### ALL TESTS ###")

#test_ANN = ANN()
test_DTree = Dtree()
#test_KNN = KNN(10)
#test_LogR = LogR()
ensemble_results=[]
individual_results=[]
x_train, x_test, y_train, y_test = test_DTree.split_data(inputs, outputs, .25)
meta= Meta_Learner("All-other-models-results-part1.csv", x_train,y_train)
#meta.Dtree()
meta.train(x_train,y_train)
#print("Naive Ensemble Test accuracy: test data set",meta.report_individual_accuracy(x_test,y_test))
#print("Naive Ensemble Test accuracy: train data set",meta.report_individual_accuracy(x_train,y_train))
print "FINAL: test data",meta.report_accuracy(x_test,y_test)
#print "FINAL: training data",meta.report_accuracy(x_train,y_train)

# print "Final Results: ", ensemble_results, individual_results
# print "Standard deviation", np.std(ensemble_results)
# print "Variance", np.var(ensemble_results)
