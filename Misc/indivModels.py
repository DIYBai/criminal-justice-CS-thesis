import RiskParser as rp
from ANN import ANN
from Dtree import Dtree
from KNN import KNN
from LogR import LogR


inputs, outputs = rp.parse_data("half_half.csv")
print(inputs)
print(outputs)

test_ANN = ANN()
test_Dtree = Dtree() #TEMP
test_KNN = KNN(10) #TEMP
test_LogR = LogR() #TEMP
x_train, x_test, y_train, y_test = test_ANN.split_data(inputs, outputs, .25)

test_ANN.train(x_train, y_train)
test_Dtree.train(x_train, y_train)
test_KNN.train(x_train, y_train)
test_LogR.train(x_train, y_train)

test_models = [ (test_ANN, "ANN"), (test_Dtree, "Dtree"), (test_KNN, "KNN"), (test_LogR, "LogR")]
for model_tup in test_models:
    print("\n###", model_tup[1], "Test ###")
    model = model_tup[0]

    zeroes = 0
    ones = 0
    false_n = 0
    false_p = 0
    true_n = 0
    true_p = 0
    for i in range(len(x_test)):
        p = model.predict([x_test[i]])
        # = testpA
        if p == 0.0:
            zeroes += 1
            if y_test[i] == 0.0:
                true_n += 1
            else:
                false_n += 1
        elif p == 1.0:
            ones += 1
            if y_test[i] == 1.0:
                true_p += 1
            else:
                false_p += 1
        else:
            print("Special val guessed: ", p)
    print("ZEROES:\t", zeroes / len(x_test), "\tfalse negative: ", false_n / (false_n + true_n + 1))
    print("ONES:  \t", ones   / len(x_test), "\tfalse positive: ", false_p / (false_p + true_p + 1))
    print("Accuracy: ", (true_n + true_p) / len(x_test))
