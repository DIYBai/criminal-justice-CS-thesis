import numpy as np
import pandas as pd
import Tester
import os
import sys
lib_path = os.path.abspath( os.path.join('..', 'ComponentModels') )
sys.path.append(lib_path)
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR

class Meta_Learner(Tester.Tester):
    def __init__(self,f_name, first_c, last_c):
        self.filename = f_name
        self.firstInpCol = first_c
        self.lastInpCol = last_c
        self.inputs, self.outputs=self.parse_data(self.filename,self.firstInpCol,self.lastInpCol)

    def run_trials(self, model,iterations):
           return
