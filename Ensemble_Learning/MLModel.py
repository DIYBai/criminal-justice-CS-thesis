from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod
import mglearn

class MLModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, x_train, y_train):
        return

    @abstractmethod
    def report_accuracy(self, x_test, y_test):
        return

    def split_data(self, inputs, outputs, test_size=.25, ):
        return train_test_split( inputs, outputs, test_size=test_size)

    # # TODO: Implemented the math, but nothing else so WIP
    # def run_trials(self):
    #     trial_n = 10
    #     accuracies = []
    #
    #     avg = 0.0
    #     for i in range(trial_n):
    #         self.split_data()
    #         self.train()
    #         acc = self.report_accuracy()
    #         accuracies.append( acc )
    #         avg += acc
    #     avg /= trial_n
    #
    #     std_dev = 0.0
    #     for i in range(trial_n):
    #         std_dev += Math.pow( accuracies[i]-avg, 2 )
    #     std_dev /= trial_n
    #     std_dev = Math.sqrt(std_dev)