import RiskParser as rp
import os
import sys
lib_path = os.path.abspath( os.path.join('..', 'ComponentModels') )
sys.path.append(lib_path)
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR
from NaiveEnsemble import NaiveEnsemble
import numpy as np
from Meta_Learner import Meta_Learner
from MetaLearnerTester import Meta_Learner_Tester

meta_tester =Meta_Learner_Tester()
inputs,outputs = meta_tester.parse_data("../Data/broward_norm.csv",3,10)
output_file = "Long/Final/broward_with_race_forposter_component_model_results_ANNs.csv"
meta_learner = Meta_Learner(output_file)
meta_tester.run_trials("broward_norm.csv",meta_learner,5,inputs, outputs)
