import csv
my_data=[]
my_processed_data=[]
test_sample=[]
input_name = 'for_testing.csv'
output_name = 'RiskAssessData.csv'

def read_f():
    my_file= open(input_name, 'rb')
    #print my_file[22], " TEST"
    my_reader= csv.reader(my_file)
    #header= my_reader.next()
    #print header
    for row in my_reader:
        my_data.append(row)
    my_processed_data.append(my_data[0])
    for i in range(1,len(my_data)):
        add=True
        for j in range(0,len(my_data[i])):
            if "Incomplete" in my_data[i][j] or "Unscored" in my_data[i][j] or 'I' in my_data[i][j] or my_data[i][j]=='':
                add=False
        if add:
            my_processed_data.append(my_data[i])
    my_file.close()
    print len(my_processed_data), " processed_data"

def write_f():
    my_new_file= open(output_name,'wb')
    wr = csv.writer(my_new_file, dialect='excel')
    wr.writerows(my_processed_data)
    my_new_file.close()

def construct_sample():
    for i in range( 1,len(my_processed_data) ):
        temp=[]
        for j in range (1, 40):
            temp.append(my_processed_data[0][j] + ' = ' + my_processed_data[i][j])
        test_sample.append(temp)
#read_f()
#write_f()
#construct_sample()
#print test_sample[0]
