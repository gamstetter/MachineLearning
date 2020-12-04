#!/usr/bin/env python2
"""
@file Neural network comparison on sigmoid and hyperbolic tangent tranfer functions

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
import argparse
import pandas as pd
import numpy as np
from random import random
from math import exp
from sklearn.model_selection import train_test_split


"""
Using this as a placeholder for future data stuff....
"""
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
        self.num_inputs = len(self.data.columns) # total number of features
        
        self.num_classes = len(set([row[-1] for row in self.data]))
        hidden_layer = [Neuron(self.num_inputs, transfer_func) for _ in range(self.num_hidden)]
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

    def forward_propogate(self, row):
        """
        @brief Calculate the output of each layer and use it as the input to the next layer

        @parameter row: The initial inputs to the network

        @return The outputs from the last layer
        """
        inputs = row
        for layer in self.layers:
            previous_layer_outputs = []
            for neuron in layer:
                active_value = self.activate(neuron.weights, inputs)
                neuron.output = neuron.transfer(active_value)
                previous_layer_outputs.append(neuron.output)

            # the input to the next layer is the output of the previous layer
            inputs = previous_layer_outputs

        # this would be the inputs to the next layer, but it's the last layer so it's actually the final outputs
        return inputs

    def activate(self, weights, inputs):
        """
        @brief Perfrom the function summation(w_i * input_i) + w0

        @param weights: The weight for each feature
        @param inputs: The feature value

        @return: The summation
        """
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation +=  (weights[i] * inputs[i])
        return activation


def sigmoid(x):
    """
    @brief Represents the sigmoid activation function

    @param x: The value to perform the sigmoid on

    @return The sigmoid of x
    """
    return 1.0 / (1.0 + exp(-x))


def hyperbolic_tangent(x):
    """
    @brief Represents the hyperbolic tangent activation function

    @param x: The value to perfrom the activation function on

    @return: The activation function applied to x
    """
    return (exp(x) - exp(-x)) / exp(x) + exp(-x)


if __name__ == "__main__":
    # import the data from the provided csv
    parser = argparse.ArgumentParser(description="Neural network for detecting bank note fraud")
    parser.add_argument("csv_path", help="Path to the csv file containing the bank note data")
    args = parser.parse_args()
    data_set = pd.read_csv(args.csv_path)
    data_set.columns = ["x1", "x2", "x3", "x4", "label"]

    # split data into test, training, and validation
    df_train, df_test = train_test_split(data_set, test_size=.33, random_state=5)
    df_train, df_validate = train_test_split(df_train, test_size=.5, random_state=5)
    np.random.seed(1)
    

    # TODO loop around this counting down the number of features and using that as the number of hidden neurons
    network_sigmoid = Network(data, 3, False, sigmoid)
    network_tan = Network(data, 3, False, hyperbolic_tangent)
    network_sigmoid.train_network(0.5, 100)
    network_tan.train_network(0.5, 100)