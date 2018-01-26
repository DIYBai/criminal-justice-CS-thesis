import random
import numpy as np

class ANNLayer:
    def __init__(self, layer_size, prev_size):
        self.size = layer_size
        self.weights = np.random.random( (prev_size, layer_size) )

    # def output(self, in_vec): #return the vector representing the outputs of each node, including bias 'node'
    #     out_vec = #TO DO: empty np array
    #     for i in range(self.size):
    #         dot_prod = np.dot(in_vec, self.weights[i])
    #         activation_out = 1 / ( 1 + math.pow(math.e, -dot_prod) )
    #         np.append(out_vec, activation_out)
    #     np.append(out_vec, 1.0)
    #     out_vec = np.array( ??? )
    #     return out_vec

    #TODO: in accordance to this SO (https://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-matrix-in-numpy),
    #      maybe make full output vector then append
    def output(self, in_vec): #return the vector representing the outputs of each node, including bias 'node'
        out_vec = []
        for i in range(self.size):
            dot_prod = np.dot(in_vec, self.weights[i])
            activation_out = 1 / ( 1 + math.pow(math.e, -dot_prod) )
            out_vec.append(activation_out)
        out_vec.append(1.0)
        out_vec = np.array(out_vec)
        return out_vec

class ANN:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        (self.layers).append(layer)

    def getOut(self, input_v):
        for i in range(len(layers)):
            input_v = layers[i].output(input_v)
        np.delete(input_v, len(input_v)-1, 0)
        return input_v

    #TODO: currently unimplemented
    def backprop:
        return 0
