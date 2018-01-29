import DecisionTree
#moduel can be found here: https://pypi.python.org/pypi/DecisionTree/3.4.3
#API of the module here: https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html#7
import loadfile as f
f.read_f()
f.write_f()
f.construct_sample()
training_datafile = "for_testing.csv"
dt = DecisionTree.DecisionTree(
                training_datafile = training_datafile,
                csv_class_column_index = 39,
                csv_columns_for_features = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38],
                entropy_threshold = 0.001,
                max_depth_desired = 20,
                symbolic_to_numeric_cardinality_threshold = 100,
                csv_cleanup_needed = 1,
     )
dt.get_training_data()
dt.calculate_first_order_probabilities()
dt.calculate_class_priors()
dt.show_training_data()
root_node = dt.construct_decision_tree_classifier()
#root_node.display_decision_tree("   ")
sample=f.test_sample[102][:38]
print f.test_sample[102]
classification = dt.classify(root_node, sample)
print classification['Rearrest=0']
print classification['Rearrest=1']
print f.test_sample[102][38]
zero=classification['Rearrest=0']
one=classification['Rearrest=1']
result=f.test_sample[102][38].split(' ')
def test_accuracy():
    min_v=2003
    max_v=10000
    correct=0
    for i in range(min_v,max_v):
        sample=f.test_sample[i][:38]
        try:
            classification = dt.classify(root_node, sample)
            zero=classification['Rearrest=0']
            one=classification['Rearrest=1']
            result=f.test_sample[102][38].split(' ')
            if round(float(zero),0)==1 and float(result[2])==0:
                correct+=1
                #print "True: 0"
            elif round(float(one),0)==1 and float(result[2])==1:
                correct+=1
                #print "True: 1"
            #else:
                #print "Possible Errors: ", classification, " Actual: ", f.test_sample[i][38]
        except AttributeError:
            print "Problem: ", sample
    accuracy=float(correct)/float(max_v-min_v)
    print "Total Accuracy: ",accuracy


test_accuracy()
