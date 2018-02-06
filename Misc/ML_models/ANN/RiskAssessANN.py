from sklearn.datasets import load_iris 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.core.display import display
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split 

def Parse_data():
	riskAsses = pd.read_csv("RiskAssessData.csv", header = 0)
	original_headers = list(riskAsses.columns.values)
	riskAsses = riskAsses._get_numeric_data()
	return riskAsses.as_matrix()

"""
def Print_results(alph, act, early_stp, hidden_lyr):
	print("\n###Test Results###")
	print("Alpha = {:4f}, Activation = {:7}, Early Stopping = {:5}".format(alph, act, early_stp)
    print("Hidden Layer Config = {:12}, Accuracy = {:7f}".format(hidden_lyr)))   
"""


class ANNTester:

    def __init__(self, model, inputs, output):
    	self.model = model
    	self.inputs = inputs
    	self.output = output

    def get_fit(self):
        self.model.fit(self.x_train, self.y_train)
        return self.model.score(self.x_test, self.y_test)

    def get_fit_avg(self,i=5,test_size=.25):
    	scores = []
    	for j in range(i):
    		self.split(test_size)
    		scores.append(self.get_fit())
    	return sum(scores)/len(scores)

    def split(self, test_size):  	
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
               self.inputs, self.output, test_size=test_size)




RA_array = Parse_data()
RA_race = RA_array[:,1]
RA_inputs = RA_array[:,4:33]
RA_output = RA_array[:,37]

mlp = MLPClassifier(hidden_layer_sizes=(10,2,))
tester = ANNTester(mlp, RA_inputs, RA_output)



print("Alpha = .0001, Activation = relu, Hidden Layer Config = (10,2) Fitness = "+ str(tester.get_fit_avg()))

tester.alpha = 1
print("Alpha = 1    , Activation = relu, Hidden Layer Config = (10,2) Fitness = " + str(tester.get_fit_avg()))

tester.alpha = .002
tester.hidden_layer_sizes = (10)
print("Alpha = .002    , Activation = relu, Hidden Layer Config = (10) Fitness = " + str(tester.get_fit_avg()))

tester.Activation = 'logistic'
print("Alpha = .002    , Activation = logistic, Hidden Layer Config = (10) Fitness = " + str(tester.get_fit_avg()))

tester.hidden_layer_sizes = (10,2)
print("Alpha = .002    , Activation = logistic, Hidden Layer Config = (10) Fitness = " + str(tester.get_fit_avg()))




