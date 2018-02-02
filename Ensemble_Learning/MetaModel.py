from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod
import mglearn

class MetaModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, x_train, y_train):
        return

    @abstractmethod
    def report_accuracy(self, x_test, y_test):
        return

    @abstractmethod
    def predict(self, x_test):
    	return

    def split_data(self, inputs, outputs, test_size=.25, ):
        return train_test_split( inputs, outputs, test_size=test_size)