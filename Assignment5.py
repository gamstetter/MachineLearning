#!/usr/bin/env python2
"""
@file Neural network comparison on sigmoid and hyperbolic tangent tranfer functions

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
import pandas as pd
import numpy as np
from random import random
from math import exp

"""
Using this as a placeholder for future data stuff....
"""
class Data:
    def __init__(self):
        self.content = pd.read_csv("data_banknote_authentication.txt")

class Neuron(object):
    def __init__(self, n_inputs, transfer_func=None):
        """
        @brief Represent one neuron in a nural network

        @param n_inputs: The number of inputs to this neuron
        @param transfer_func: The function to linearlize the data
        """
        self.transfer = transfer_func

        # each neuron will add a bias term to the number of inputs
        self.weights = np.random.rand(n_inputs + 1, 1)

class Network:
    """
    Only going to write this code for the banknote code.
    """
    def __init__(self, data, num_hidden, transfer_func):
        """
        @brief Represent a neural netowrk as a colletion of rows of neurons
        
        @param data: The training data
        @param num_hidden: The number of neurons in the hidden layer
        @param transfer_func: The transfer function that the hidden layer neurons should use
        """
        self.num_hidden = num_hidden
        self.data = data.content.to_numpy()
        num_inputs = len(self.data[0])
        
        self.num_classes = len(set([row[-1] for row in self.data]))
        hidden_layer = [Neuron(num_inputs, transfer_func) for _ in range(self.num_hidden)]
        output_layer = [Neuron(len(hidden_layer)) for _ in range(self.num_classes)]

        self.layers = [hidden_layer, output_layer]
    
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
    np.random.seed(1)
    network = Network(data, 3, False)
    network.train_network(0.5, 100)