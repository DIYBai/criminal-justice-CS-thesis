import RiskParser as rp
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR
from NaiveEnsemble import NaiveEnsemble
from discreteSL import discreteSL
from fullSL import fullSL



inputs, outputs = rp.parse_data()

test_DTree = Dtree()
test_KNN = KNN(50)
test_LogR = LogR()
test_ANN = ANN()
x_train, x_test, y_train, y_test = test_ANN.split_data(inputs, outputs, .25)


##print("### Naive Ensemble ###")
##test_NaiveEnsemble = NaiveEnsemble([test_ANN, test_DTree, test_KNN, test_LogR])
##test_NaiveEnsemble.train(x_train, y_train)
##print("Accuracy: " + str(test_NaiveEnsemble.report_accuracy(x_test,y_test)))


##print("### Discrete Super Learner Ensemble ###")
##test_discreteSL = discreteSL([test_ANN, test_DTree, test_KNN, test_LogR])
##test_discreteSL.train(x_train, y_train)
##print("Accuracy: " + str(test_discreteSL.report_accuracy(x_test,y_test)))

print("### Full Super Learner Ensemble ###")
test_fullSL = fullSL([test_ANN, test_DTree, test_KNN, test_LogR])
test_fullSL.train(x_train, y_train)
print("Accuracy: " + str(test_fullSL.report_accuracy(x_test,y_test)))