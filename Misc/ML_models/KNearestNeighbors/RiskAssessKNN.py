from sklearn.datasets import load_iris 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.core.display import display
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 



class KNN:

    def __init__(self, inputs, output, original_headers):
    	self.inputs = inputs
    	self.output = output
    	self.original_headers = original_headers

    def get_fit(self,k):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.x_train, self.y_train)
        return knn.score(self.x_test, self.y_test)

    def split(self):  	
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
               self.inputs, self.output)

riskAsses = pd.read_csv("Data/RiskAssessData.csv", header = 0)
original_headers = list(riskAsses.columns.values)
riskAsses = riskAsses._get_numeric_data()
numpy_array = riskAsses.as_matrix()

race = numpy_array[:,1]
RA_inputs = numpy_array[:,4:33]
RA_output = numpy_array[:,37]
original_headers = original_headers[4:33]

Risk_Assess_KNN = KNN(RA_inputs, RA_output, original_headers)
for i in range(1,30):
    for j in range(0,3):
        Risk_Assess_KNN.split()
        print("K =" + str(i) +  " Fitness = " + str(Risk_Assess_KNN.get_fit(i)))




