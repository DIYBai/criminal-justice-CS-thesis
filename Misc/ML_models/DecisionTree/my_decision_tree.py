import numpy as np
class D_tree:
    def __init__(self,size,columns):
        self.height=10
        self.records=size
        self.data=[]
        self.processed_data=[]
        self.column=columns
        self.yes=0.0
        self.no=0.0
        self.set_entropy=-1.0
    def load_file(self):
        my_file= open('RiskAssessData.csv', 'r')
        my_line= my_file.readline()
        my_data=[]
        #print my_file[22], " TEST"
        counter=0
        for line in my_file:
            if "Incomplete" not in line and "Unscored" not in line and 'I' not in line:
                my_list=line.split(',')
                for i in range(0,self.column):
                #if my_list[i]!='' and my_list[i]!= "Incomplete" and my_list[i]!= "Unscored" and my_list[i]!= 'I':
                    if my_list[i]!='':
                        my_list[i]=float(my_list[i])
                            # if isinstance(my_list[i],float):
                            #     a=2
                            # else:
                            #     print "Test: False ", my_list[i], " counter: ", counter
                            #     counter+=1
                my_data.append(my_list)
        print len(my_data)
        my_file.close()
        self.data=my_data
    def process_data(self):
        x=np.array([[1,2,3],[4,5,6],[7,8,9]])
        print np.swapaxes(x,0,1)


    #this method counts the number of yes and no for the target column/clasification
    def classify_target(self):
        self.yes,self.no=self.count_yes_no(39)
        print self.yes, " Confirm"
        print self.no, "Confirm"

    #this method simply count the number of yes and no of a given column
    def count_yes_no(self,column):
        no=0
        yes=0
        for i in range(0,self.records):
            #print " Record: ", self.data[i][39]
            if self.data[i][column]==0.0:
                no+=1.0
            elif self.data[i][column]==1.0:
                yes+=1.0
            else:
                print "Error"
        print "Yes: ", yes," No: ", no
        return (yes,no)
    def better_count(self):
        x=np.swapaxes(self.data[:40],0,1)
        val, counts = np.unique(x, return_counts=True)
        print val,counts
    def initialize(self):
        self.set_entropy=-(self.yes/float(self.records))*np.log2(self.yes/float(self.records))-(self.no/float(self.records))*np.log2(self.no/float(self.records))
        print self.set_entropy



if __name__ == "__main__":
    my_tree= D_tree(40,40)
    my_tree.load_file()
    my_tree.classify_target()
    #my_tree.process_data()
    #my_tree.initialize()
    #my_tree.better_count()
