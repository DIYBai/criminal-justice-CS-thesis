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
vectorIn, vectorout = rp.parse_data(vector_file, 0, 29)

# print(len(inputs[0]))
# print(len(vectorIn[0]))
# print(inputs)
# print(outputs)

test_ANN = ANN()
test_DTree = Dtree()
test_KNN = KNN(10)
test_LogR = LogR()
test_NaiveEnsemble = NaiveEnsemble( [test_ANN, test_KNN, test_DTree, test_LogR] )
model_names = ["ANN", "KNN", "Dtree", "LogR",]

x_train, x_test, y_train, y_test = test_DTree.split_data(inputs, outputs, .05)
test_NaiveEnsemble.train(x_train, y_train)

model_n = 0
text = open("probs.txt", "w")
for model in test_NaiveEnsemble.model_list:
    model_probs = []
    for i in range(len(vectorIn)):
        prob = model.predict_prob( [vectorIn[i]] )
        model_probs.append( (prob[0][0], i) )
    model_probs.sort()

    text.write( "{:s}:".format(model_names[model_n]) )
    for i in range(len(model_probs)):
        text.write("\n" + str(model_probs[i]))
    text.write("\n\n")

    model_n += 1
text.close()
