import RiskParser as rp
from ANN import ANN
from NaiveEnsemble import NaiveEnsemble



inputs, outputs = rp.parse_data()
print(inputs)
print(outputs)

print("### ANN Test ###")
test_ANN = ANN()
x_train, x_test, y_train, y_test = test_ANN.split_data(inputs, outputs, .25)
##test_ANN.train(x_train, y_train)
##print("Accuracy: " + str(test_ANN.report_accuracy(x_test,y_test)))

test_DTree = ANN()



test_NaiveEnsemble = NaiveEnsemble([test_ANN, test_DTree])
test_NaiveEnsemble.train(x_train, y_train)
print(test_NaiveEnsemble.report_accuracy(x_test,y_test))