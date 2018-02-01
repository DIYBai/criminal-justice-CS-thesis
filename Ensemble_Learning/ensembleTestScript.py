import RiskParser as rp
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR
from NaiveEnsemble import NaiveEnsemble



inputs, outputs = rp.parse_data("half_half.csv")
print(inputs)
print(outputs)
print("### ALL TESTS ###")

test_ANN = ANN()
test_DTree = Dtree()
test_KNN = KNN(10)
test_LogR = LogR()
x_train, x_test, y_train, y_test = test_DTree.split_data(inputs, outputs, .25)




test_NaiveEnsemble = NaiveEnsemble([test_ANN, test_LogR, test_DTree, test_KNN])
test_NaiveEnsemble.train(x_train, y_train)
#print("Naive Ensemble Test accuracy: ",test_NaiveEnsemble.report_accuracy(x_test,y_test))
print("Naive Ensemble Test accuracy: ",test_NaiveEnsemble.report_accuracy(x_train,y_train))

#test_KNN.train(x_train,y_train)
#test_NaiveEnsemble.model_list.append(test_KNN)
print "Individual accuracy: ", test_NaiveEnsemble.report_individual_accuracy(x_test,y_test)
print "Individual accuracy: ", test_NaiveEnsemble.report_individual_accuracy(x_train,y_train)
