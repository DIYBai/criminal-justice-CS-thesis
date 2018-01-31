import RiskParser as rp
from ANN import ANN
from Dtree import Dtree


inputs, outputs = rp.parse_data()
print(inputs)
print(outputs)

print("### ANN Test 1 ###")
test_ANN = ANN()
x_train, x_test, y_train, y_test = test_ANN.split_data(inputs, outputs, .25)
test_ANN.train(x_train, y_train)
for i in range(len(x_test)):
    prediction = test_ANN.predict([x_test[i]])
    if prediction != 0.0:
        print(prediction)
print("Accuracy: " + str(test_ANN.report_accuracy(x_test,y_test)))

inputs, outputs = rp.parse_data("testData.csv")
print(inputs)
print(outputs)

print("### ANN Test 2 ###")
test_ANN = ANN()
x_train, x_test, y_train, y_test = test_ANN.split_data(inputs, outputs, .25)
test_ANN.train(x_train, y_train)
for i in range(len(x_test)):
    prediction = test_ANN.predict([x_test[i]])
    if prediction != 0.0:
        print(prediction)
print("Accuracy: " + str(test_ANN.report_accuracy(x_test,y_test)))


# print("### Dtree Test ###")
# test_Dtree = Dtree()
#x_train, x_test, y_train, y_test = test_Dtree.split_data(inputs, outputs, .25)
#test_Dtree.train(x_train, y_train)
# test_Dtree.run_trials(inputs, outputs, .25)
#print("Accuracy: " + str(test_Dtree.report_accuracy(x_test,y_test)))
