import sys

#should work for both individual models and ensemble models
def test_model(model, inp, out):
    entries = len(inp)

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
            if out[i] == 0.0:
                true_n += 1
            else:
                false_n += 1
        elif p == 1.0:
            ones += 1
            if out[i] == 1.0:
                true_p += 1
            else:
                false_p += 1
        else:
            print("Special val guessed:", p)

    zero_rate = zeroes / entries
    one_rate =  ones   / entries
    accuracy = (true_n + true_p) / entries
    print("ZEROES:\t",  "{:6.4f}".format(zero_rate), "(", zeroes, ")\tFalse Negative: ", false_n / (false_n + true_n + 0.00001))
    print("ONES:  \t",  "{:6.4f}".format(one_rate),  "(", ones,   ")\tFalse Positive: ", false_p / (false_p + true_p + 0.00001))
    print("Accuracy: ", "{:6.4f}".format(accuracy), "\n")
    return [len(inp), accuracy, ( false_n / (false_n + true_n + 0.00001) ), ( false_p / (false_p + true_p + 0.00001) )]

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

def get_distances(inp):#, min_cutoff = 0.0, max_cutoff = sys.maxsize):
    distances = []
    min_dist = euclid_dist_squared(inp[0], inp[1])
    max_dist = euclid_dist_squared(inp[0], inp[1])
    avg = 0
    count = 0
    for i in range(len(inp)):
        for j in range(i + 1, len(inp)):
            dist = euclid_dist_squared(inp[i], inp[j])
            avg += dist
            count += 1
            if dist > max_dist:
                max_dist = dist
            elif dist < min_dist:
                min_dist = dist
            #if min_cutoff <= dist and dist <= max_cutoff:
            distances.append( (dist, i, j) )
    distances.sort()
    avg = avg / count
    return distances, min_dist, max_dist, avg

def euclid_dist_squared(v1, v2):
    dist = 0
    for i in range(len(v1)):
        dist += (v1[i] - v2[i])**2
    return dist

def get_OR(inp, out): #odds ratio (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2938757/)
    odds_ratios = []
    print(out)
    for i in range(len(inp[0])): #for each columns
        #print("Column:", i)
        exp_c = 0
        exp_n = 0
        unexp_c = 0
        unexp_n = 0
        for j in range(len(inp)):
            #print(inp[j][i], out[j])
            if inp[j][i] == out[j]:
                if out[j] == 0:
                    unexp_n += 1
                else:
                    exp_c += 1
            else:
                if out[j] == 0:
                    exp_n += 1
                else:
                    unexp_c += 1
        odds_ratio = (exp_c*unexp_n)/(unexp_c*exp_n + 0.000001)
        risk_ratio = ( exp_c*(unexp_n+unexp_c) ) / ( unexp_c*(exp_n+exp_c) + 0.000001 )
        odds_ratios.append( (odds_ratio, risk_ratio, i) )
        #print("\n")
    odds_ratios.sort()
    return odds_ratios
