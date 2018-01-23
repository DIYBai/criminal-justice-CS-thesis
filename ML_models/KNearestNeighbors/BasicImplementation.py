class data_pt:
    values = []
    label = ""

    def assignValues(new_values):
        values = new_values

    def assignLabel(new_label):
        label = new_label

def euclidDist(vals1, vals2): #really computes square of eulcidean distance
    # assert len(vals1) == len(vals2) #dimensionality should match
    dist = 0
    for i in range(len(vals1)):
        dist += pow(vals1[i] - vals2[i], 2)
    return dist

def distanceSortedList(new_datum, data_list):
    out_list = []
    for i in range(len(data_list)):
        curr_datum = data_list[i]
        dist = euclidDist(new_datum.values, curr_datum.values)
        out_list.append( (dist, curr_datum) ) #maybe just store distance and label? might not need whole data_pt obj
    out_list.sort
    print(out_list) #Temporary, for testing

def classify(k, new_datum, data_list):
    sorted_data = distanceSortedList(new_datum.values, data_list)
    label_dict = {}
    for i in range(k):
        curr_label = sorted_data[i].label
        if (curr_label in label_dict):
            label_dict[curr_label] += 1
        else:
            label_dict[curr_label] = 1
    key_list = label_dict.keys()
    if len(key_list) > 1:
        sorted_labels = sorted(key_list)
        freq_labels = [sorted_labels[0]]
        idx = 1
        while( label_dict[ sorted_labels[idx] ] == label_dict[ sorted_labels[0] ] and idx < len(sorted_labels) ):
            freq_labels.append(sorted_labels[idx])
            idx += 1
        random_idx = random.randrange(0, len(freq_labels))
        return freq_labels[random_idx]
    else:
        return key_list[0]
