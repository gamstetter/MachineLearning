import pandas as pd
from random import random
from math import exp

"""
Using this as a placeholder for future data stuff....
"""
class Data:
    def __init__(self):
        self.content = pd.read_csv("data_banknote_authentication.txt")

class Network:
    """
    Only going to write this code for the banknote code.
    """
    def __init__(self, data, num_hidden, is_sigmoid_squash):
        """
        Instantiate the data that we need for future operations.
        """
        self.num_hidden = num_hidden
        self.is_sigmoid_squash = is_sigmoid_squash
        self.data = data.content.to_numpy()
        
        self.output_amount = len(set([row[-1] for row in self.data]))
        neural_network = []
        hidden_layer = [{'weights':[random() for i in range(len(self.data[0]) + 1)]} for i in range(self.num_hidden)]
        neural_network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(len(self.data[0]) + 1)]} for i in range(self.output_amount)]
        neural_network.append(output_layer)

        self.neural_network = neural_network
    
    def train_network(self, learning_rate, len_epochs):
        """
        Use the data in the class to train us.
        """
        for epoch in range(0, len_epochs, 1):
            total_error = 0
            for row in self.data:
                forward_result = self.forward_propogate(row)
                expected = [0 for i in range(self.output_amount)]
                #expected[row[-1]] = 1
                total_error += sum([(expected[i]-forward_result[i])**2 for i in range(len(expected))])

    def transfer(self, activation):
	    return 1.0 / (1.0 + exp(-activation))

    def forward_propogate(self, row):
        temp_row = row
        for i in self.neural_network:
            l = []
            for j in i:
                active_value = self.activate(j['weights'], row)
                j['output'] = self.transfer(active_value)
                l.append(j['output'])
            temp_row = l
        return temp_row

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation =  activation + (weights[i] * inputs[i])
        return activation

if __name__ == "__main__":
    data = Data()
    # TODO data needs split before here.
    network = Network(data, 3, False)
    network.train_network(0.5, 100)