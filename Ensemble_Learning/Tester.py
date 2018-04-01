from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod
import RiskParser as rp
import mglearn

class Tester:
    __metaclass__ = ABCMeta

    def parse_data(self, filename, first, last):
        return rp.parse_data(filename=filename,firstInpCol = first, lastInpCol = last)

    @abstractmethod
    def run_trials(self, model,iterations, inputs, outputs):
       return

    @abstractmethod
    def export_results(self):
        return
