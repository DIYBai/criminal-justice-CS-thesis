import RiskParser as rp
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR
from NaiveEnsemble import NaiveEnsemble
import numpy as np

in_file = "RiskAssessData.csv"
in_file = "half_half.csv"
inputs, outputs = rp.parse_data(in_file)
# print(inputs)
# print(outputs)
print("### ALL TESTS ###")

test_ANN = ANN()
test_DTree = Dtree()
test_KNN = KNN(10)
test_LogR = LogR()
ensemble_results = []
individual_results = []

for a in range(3):
    print("########\nTrial", a, "\n########")
    x_train, x_test, y_train, y_test = test_DTree.split_data(inputs, outputs, .25)
    test_NaiveEnsemble = NaiveEnsemble( [test_ANN, test_KNN, test_LogR, test_DTree] )
    test_NaiveEnsemble.train(x_train, y_train)
    model_names = ["ANN", "KNN", "LogR", "Dtree"]

    ensemble = test_NaiveEnsemble.report_accuracy(x_test,y_test)
    print("Naive ensemble accuracy:", ensemble) #test data set
    ensemble_results.append(ensemble)

    individual = test_NaiveEnsemble.report_individual_accuracy(x_test, y_test)
    print("Individual accuracy:", individual) #test data set
    individual_results.append(individual)

    ######################################
    for i in range(4):
        print("- Testing", model_names[i], "-")
        model = test_NaiveEnsemble.model_list[i]
        zeroes = 0
        ones = 0
        false_n = 0
        false_p = 0
        true_n = 0
        true_p = 0
        # for i in range(len(x_test)):
        #     p = model.predict([x_test[i]])
        #     # = testpA
        #     if p == 0.0:
        #         zeroes += 1
        #         if y_test[i] == 0.0:
        #             true_n += 1
        #         else:
        #             false_n += 1
        #     elif p == 1.0:
        #         ones += 1
        #         if y_test[i] == 1.0:
        #             true_p += 1
        #         else:
        #             false_p += 1
        #     else:
        #         print("Special val guessed: ", p)
        # print("ZEROES:\t", zeroes / len(x_test), "\tfalse negative: ", false_n / (false_n + true_n + 1))
        # print("ONES:  \t", ones   / len(x_test), "\tfalse positive: ", false_p / (false_p + true_p + 1))
        # print("Accuracy: ", (true_n + true_p) / len(x_test), "\n"
        for i in range(len(x_train)):
            p = model.predict([x_train[i]])
            # = testpA
            if p == 0.0:
                zeroes += 1
                if y_train[i] == 0.0:
                    true_n += 1
                else:
                    false_n += 1
            elif p == 1.0:
                ones += 1
                if y_train[i] == 1.0:
                    true_p += 1
                else:
                    false_p += 1
            else:
                print("Special val guessed: ", p)
        print("ZEROES:\t", zeroes / len(x_train), "\tfalse negative: ", false_n / (false_n + true_n + 1))
        print("ONES:  \t", ones   / len(x_train), "\tfalse positive: ", false_p / (false_p + true_p + 1))
        print("Accuracy: ", (true_n + true_p) / len(x_train), "\n")
    ######################################

# print("\n\nFinal Results:", ensemble_results, individual_results)
print("\n\nStandard deviation", np.std(ensemble_results))
print("Variance", np.var(ensemble_results))
