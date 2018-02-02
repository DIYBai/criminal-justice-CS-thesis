
#should work for both individual models and ensemble models
def test_model(model, inp, out):
    entries = entries

    zeroes = 0
    ones = 0
    false_n = 0
    false_p = 0
    true_n = 0
    true_p = 0

    for i in range(entries):
        p = model.predict([inp[i]])
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
            print("Special val guessed:", p)

    zero_rate = zeroes / entries
    one_rate =  ones   / entries
    accuracy = (true_n + true_p) / entries
    print("ZEROES:\t",  "{:6.4f}".format(zero_rate), "(", zeroes, ")\tFalse Negative: ", false_n / (false_n + true_n + 1))
    print("ONES:  \t",  "{:6.4f}".format(one_rate),  "(", ones,   ")\tFalse Positive: ", false_p / (false_p + true_p + 1))
    print("Accuracy: ", "{:6.4f}".format(accuracy), "\n")

#TODO: make it pretty?
def test_ensemble_complete(ensemble, inp, out):
    for i in ensemble.model_list:
        test_model(i, inp, out)
    test_model(ensemble, inp, out)

#TODO: calculate standard deviation and average
#TODO: low priority: format for prettiness
def run_trials(model, inp, out, k = 10):
    for i in range(k):
        test_model(model, inp, out)
