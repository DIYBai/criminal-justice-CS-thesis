import RiskParser as rp
import os
import sys
from NaiveEnsemble import NaiveEnsemble
from discreteSL import discreteSL
from ensembleTester import ensembleTester
from annSL import annSL
from fullSL import fullSL
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
lib_path = os.path.abspath(os.path.join('..','ComponentModels'))
sys.path.append(lib_path)
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR



inputs, outputs = rp.parse_data("../Data/broward_norm.csv",4,10,13)

test_DTree = Dtree()
test_KNN = KNN(3)
test_LogR = LogR()
test_ANN = ANN()
x_train, x_test, y_train, y_test = test_ANN.split_data(inputs, outputs, .25)

##test_DTree.train(x_train,y_train)
##print(test_DTree.report_accuracy(x_test,y_test))

##print("### Naive Ensemble ###")
##test_NaiveEnsemble = NaiveEnsemble([test_ANN, test_DTree, test_KNN, test_LogR])
##test_NaiveEnsemble.train(x_train, y_train)
##print("Accuracy: " + str(test_NaiveEnsemble.report_accuracy(x_test,y_test)))


##print("### Discrete Super Learner Ensemble ###")
##test_discreteSL = discreteSL([test_ANN, test_DTree, test_KNN, test_LogR])
##test_discreteSL.train(x_train, y_train)
##print("Accuracy: " + str(test_discreteSL.report_accuracy(x_test,y_test)))

print("### ANN Super Learner Ensemble ###")
test_fullSL = annSL([test_ANN, test_DTree, test_KNN, test_LogR])
e_tester = ensembleTester([test_fullSL])
e_tester.test(inputs,outputs)
##test_fullSL.train(x_train, y_train)
##print("Accuracy: " + str(test_fullSL.report_accuracy(x_test,y_test)))