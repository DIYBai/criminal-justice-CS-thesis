import RiskParser as rp
import os
import sys
from NaiveEnsemble import NaiveEnsemble
from discreteSL import discreteSL
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
from slTester import slTester

## ---Component ML Models--- ## 
## models were selected by Meta Learner output ranking of ideal model configurations ##
a_DTree = Dtree('gini','best',3)
b_DTree = Dtree('entropy','best',3)
c_DTree = Dtree('entropy','random',3)
a_KNN = KNN(11,'uniform','auto')
b_KNN = KNN(15, 'uniform','kd_tree')
c_KNN = KNN(23, 'uniform','ball_tree')
a_LogR = LogR(algo='newton-cg', c=3)
b_LogR = LogR(algo='newton-cg', c=6)
c_LogR = LogR(algo='newton-cg', c=9)
a_ANN = ANN(iteration=8700,layers=[100])
b_ANN = ANN(iteration=2700,layers=[100,90,80,70])
inputs, outputs = rp.parse_data("../Data/broward_norm.csv",4,10,13)
model_list = [a_DTree, b_DTree, c_DTree, a_KNN, b_KNN, c_KNN, a_KNN, b_KNN, c_KNN, a_LogR, b_LogR, c_LogR, a_ANN, b_ANN]

## ---Ensemble Tester--- ##
x_train, x_test, y_train, y_test = a_ANN.split_data(inputs, outputs, .25)
test_fullSL = annSL([a_DTree, b_DTree, a_KNN, b_KNN, a_LogR, b_LogR, a_ANN, b_ANN])
sl_tester = slTester()

print("Dataset: Breast Cancer ")
print("Input: standard")
print("Predict: malignant or not")
print("------------------------------")
inputs, outputs = rp.parse_data("../Data/breast_cancer.csv",1,9,10)
sl_tester.run_trials("breast_cancer",test_fullSL,5,inputs,outputs)

print("Dataset: Broward Norm ")
print("Input: no compas desc, no race ")
print("Predict: Actual Recidivism")
print("------------------------------")
inputs, outputs = rp.parse_data("../Data/broward_norm.csv",4,9,13)
sl_tester.run_trials("broward_norm_no_compas_no_race",test_fullSL,5,inputs,outputs)

print("Dataset: Broward Norm ")
print("Input: with compas desc, no race ")
print("Predict: Actual Recidivism")
print("------------------------------")
inputs, outputs = rp.parse_data("../Data/broward_norm.csv",4,10,13)
sl_tester.run_trials("broward_norm_with_compas_no_race",test_fullSL,5,inputs,outputs)

print("Dataset: Broward Norm ")
print("Input: no compas desc, with race ")
print("Predict: Actual Recidivism")
print("------------------------------")
inputs, outputs = rp.parse_data("../Data/broward_norm.csv",3,9,13)
sl_tester.run_trials("broward_norm_no_compas__with_race",test_fullSL,5,inputs,outputs)

print("Dataset: Broward Norm ")
print("Input: with compas desc, with race ")
print("Predict: Actual Recidivism")
print("------------------------------")
inputs, outputs = rp.parse_data("../Data/broward_norm.csv",3,10,13)
sl_tester.run_trials("broward_norm_with_compas__with_race",test_fullSL,5,inputs,outputs)

print("Dataset: Broward Norm ")
print("Input: without compas desc")
print("Predict: Race")
print("------------------------------")
inputs, outputs = rp.parse_data("../Data/broward_norm.csv",4,9,3)
sl_tester.run_trials("broward_norm_no_compas_predict_race",test_fullSL,5,inputs,outputs)

print("Dataset: Broward Norm ")
print("Input: with compas desc")
print("Predict: Race")
print("------------------------------")
inputs, outputs = rp.parse_data("../Data/broward_norm.csv",4,10,3)
sl_tester.run_trials("broward_norm_with_compas__predict_race",test_fullSL,5,inputs,outputs)

