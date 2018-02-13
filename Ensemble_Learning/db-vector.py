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
# import db-vectorWriter


in_file = "../Data/RiskAssessData.csv"
inputs, outputs = rp.parse_data(in_file)
vector_file = "../Data/vectorData.csv"
vectorin, vectorout = rp.parse_data(vector_file, 0, 29)

# print(len(inputs[0]))
# print(len(vectorin[0]))
# print(inputs)
# print(outputs)

test_ANN = ANN()
test_DTree = Dtree()
test_KNN = KNN(10)
test_LogR = LogR()
test_NaiveEnsemble = NaiveEnsemble( [test_ANN, test_KNN, test_DTree, test_LogR] )
# test_NaiveEnsemble = NaiveEnsemble( [test_ANN, test_KNN, test_DTree] ) #, test_LogR])
model_names = ["ANN", "KNN", "Dtree" "LogR",]

x_train, x_test, y_train, y_test = test_DTree.split_data(inputs, outputs, .05)
test_NaiveEnsemble.train(x_train, y_train)

ANN_probs = []
KNN_probs = []
DTree_probs = []
LogR_probs = []
Average_probs = []
#prints out probability of 0
for i in range(len(vectorin)):
    print("* ROW", i, "*")
    # print( [inputs[i]] )
    # print( [vectorin[i]], "\n")
    # ensemble_pred = test_NaiveEnsemble.predict([vectorin[i]])
    ensemble_prob = test_NaiveEnsemble.predict_prob([vectorin[i]])
    # print(ensemble_prob[0])
    ANN_probs.append(   (ensemble_prob[0][0][0], i) )
    KNN_probs.append(   (ensemble_prob[1][0][0], i) )
    DTree_probs.append( (ensemble_prob[2][0][0], i) )
    LogR_probs.append(  (ensemble_prob[3][0][0], i) )
    avgs = np.average(ensemble_prob, axis = 0)
    # print("\nAverages array:", avgs, "\n\n")
    print('Average: {:.3f}\n'.format(avgs[0][0]))
    Average_probs.append(  (avgs[0][0], i) )
    # print( "\tEnsemble:", '{:.3f}'.format( ensemble_prob[0][0] ), "\n" )
ANN_probs.sort()
KNN_probs.sort()
DTree_probs.sort()
LogR_probs.sort()
Average_probs.sort()

print("ANN:", ANN_probs)
print("KNN:", KNN_probs)
print("DTree:",DTree_probs)
print("LogR", LogR_probs)
print("Naive Ensemble:", Average_probs)
