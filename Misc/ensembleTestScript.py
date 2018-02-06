import RiskParser as rp
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR
from NaiveEnsemble import NaiveEnsemble
import numpy as np



inputs, outputs = rp.parse_data("RiskAssessData.csv")
print(inputs)
print(outputs)
print("### ALL TESTS ###")

test_ANN = ANN()
test_DTree = Dtree()
test_KNN = KNN(10)
test_LogR = LogR()
ensemble_results=[]
individual_results=[]
for i in range(0,3):
    x_train, x_test, y_train, y_test = test_DTree.split_data(inputs, outputs, .25)
    test_NaiveEnsemble = NaiveEnsemble([test_ANN,test_KNN,test_LogR,test_DTree])
    test_NaiveEnsemble.train(x_train, y_train)
    ensemble=test_NaiveEnsemble.report_accuracy(x_test,y_test)
    print("Naive Ensemble Test accuracy: test data set",ensemble)
    #print("Naive Ensemble Test accuracy: train data set",test_NaiveEnsemble.report_accuracy(x_train,y_train))

    #test_KNN.train(x_train,y_train)
    #test_NaiveEnsemble.model_list.append(test_KNN)
    individual=test_NaiveEnsemble.report_individual_accuracy(x_test,y_test)
    print "Individual accuracy: test data set", individual
    #print "Individual accuracy: test train set", test_NaiveEnsemble.report_individual_accuracy(x_train,y_train)
    ensemble_results.append(ensemble)
    individual_results.append(individual)
print "Final Results: ", ensemble_results, individual_results
print "Standard deviation", np.std(ensemble_results)
print "Variance", np.var(ensemble_results)
