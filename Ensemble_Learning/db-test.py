import numpy as np
import RiskParser as rp
from helper import *
from fullSL import *

import os
import sys
model_path = os.path.abspath( os.path.join('..', 'ComponentModels') )
sys.path.append(model_path)
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR
from NaiveEnsemble import NaiveEnsemble


in_file = "../Data/vectorData.csv"
inputs, outputs = rp.parse_data(in_file, lastInpCol = 39)
# or_list = get_OR(inputs, outputs)
# for i in range(len(or_list)):
#     print(or_list[i])
# data_dists, min_dist, max_dist, avg_dist = get_distances(inputs)
# i = 0
# MIN_CUTOFF = 2.0
# while(i < len(data_dists) and data_dists[i][0] <= MIN_CUTOFF):
#     # print(data_dists[i])
#     i += 1
# print("Min:", min_dist, "\tMax:", max_dist, "\tMean:", avg_dist, "\tMedian:", (data_dists[len(data_dists)//2])[0] )
# print( "N below cutoff:", i, "out of", len(data_dists), " ({:6.4f})".format(i/len(data_dists)) )
print(len(inputs[0]))
print(inputs)
print(outputs)

print("### ALL TESTS ###")

test_ANN = ANN()
test_DTree = Dtree()
test_KNN = KNN(10)
test_LogR = LogR()
test_NaiveEnsemble = NaiveEnsemble( [test_ANN, test_KNN, test_LogR, test_DTree] )
# test_NaiveEnsemble = NaiveEnsemble( [test_ANN] )
model_names = ["ANN", "KNN", "LogR", "Dtree"]

ensemble_results = []
individual_results = []

for a in range(3):
    print("########\nTrial", a, "\n########")
    x_train, x_test, y_train, y_test = test_DTree.split_data(inputs, outputs, .05)
    test_NaiveEnsemble.train(x_train, y_train)

    ensemble = test_NaiveEnsemble.report_accuracy(x_train, y_train)
    print("Naive ensemble accuracy:", ensemble) #test data set
    ensemble_results.append(ensemble)

    individual = test_NaiveEnsemble.report_individual_accuracy(x_train, y_train)
    print("Individual accuracy:", individual) #test data set
    individual_results.append(individual)

    ######################################
    for i in range(1):#4):
        print("- Testing", model_names[i], "-")
        model = test_NaiveEnsemble.model_list[i]
        test_model(model, x_train, y_train)
    ######################################

# print("\n\nFinal Results:", ensemble_results, individual_results)
# print("\n\nStandard deviation", np.std(ensemble_results))
# print("Variance", np.var(ensemble_results))


print("\n### Full Super Learner Ensemble ###")
test_fullSL = fullSL([test_ANN, test_DTree, test_KNN, test_LogR])
test_fullSL.train(x_train, y_train)
print("Accuracy: " + str(test_fullSL.report_accuracy(x_test,y_test)))
