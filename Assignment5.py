#!/usr/bin/env python2
"""
@file Neural network comparison on sigmoid and hyperbolic tangent tranfer functions

@authors: Grace Gamstetter, Michael Gentile, and Richard Hammond
"""
import argparse
import pandas as pd
import numpy as np
from random import random
from math import tanh
from sklearn.model_selection import train_test_split


"""
Using this as a placeholder for future data stuff....
"""
class Neuron(object):
    def __init__(self, n_inputs, transfer_func=None, cost_func=None, output_layer=False):
        """
        @brief Represent one neuron in a nural network

        @param n_inputs: The number of inputs to this neuron
        @param transfer_func: The function to linearlize the data
        @param cost_func: The derivative of the transfer function
        @param output_layer: True if this is an output layer neuron, else False
        """
        self.transfer = transfer_func
        self.cost = cost_func
        self.output = None
        self.error = None
        self.delta = None

        # the hidden layer has a different error calculation than the output layer
        self.error_func = self.output_error if output_layer else self.hidden_error

        # each neuron will add a bias term to the number of inputs
        self.weights = np.random.rand(n_inputs + 1, 1)

    def activate(self, inputs):
        """
        @brief Perfrom the dot product of the input and weight vectors

        @param inputs: The feature value or activation from previoius layer

        @return: Activation vector
        """
        return np.dot(self.weights, inputs)


    def output_error(self, expected):
        """
        @brief Error calculation used for the output layer

        @param expected: The expected value of the output

        @return The error
        """
        return expected - self.output

    def hidden_error(self, out_error):
        """
        @brief The error from the output layer is taken and then multiplied by the weights of 
               the inputs to this neuron.
        
        @param out_error: The error from the output layer

        @return The error
        """
        return np.dot(self.weights, out_error)


class Network:
    """
    Only going to write this code for the banknote code.
    """
    def __init__(self, data, num_hidden, transfer_func, cost_func):
        """
        @brief Represent a neural netowrk as a colletion of rows of neurons
        
        @param data: The training data
        @param num_hidden: The number of neurons in the hidden layer
        @param transfer_func: The transfer function that the hidden layer neurons should use
        @param cost_func: The derivative of the transfer function
        """
        self.num_hidden = num_hidden
        self.data = data.content.to_numpy()
        self.num_inputs = len(self.data.columns) # total number of features
        
        self.num_classes = len(set([row[-1] for row in self.data]))
        hidden_layer = [Neuron(self.num_inputs, transfer_func, cost_func) for _ in range(self.num_hidden)]
        output_layer = [Neuron(len(hidden_layer), output_layer=True) for _ in range(self.num_classes)]

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
                active_value = neuron.activate(inputs)
                neuron.output = neuron.transfer(active_value)
                previous_layer_outputs.append(neuron.output)

            # the input to the next layer is the output of the previous layer
            inputs = previous_layer_outputs

        # this would be the inputs to the next layer, but it's the last layer so it's actually the final outputs
        return inputs

    def back_propogate(self):
        """
        @brief Staring at the last layer, send the error signal up the layers
        """
        # The input driving the error function. We two expected outputs (one per class). These kick off the back propogation.
        expected_out = [0, 1]
        for layer_num, layer in reversed(list(enumerate(self.layers))):
            for i, neuron in enumerate(layer):
                # if this is not the last layer, we need to calculate an error per neuron we are connected
                # to in the next layer
                if layer_num != len(self.layers) - 1:
                    neuron.error = 0
                    for next_neuron in self.layers[layer_num + 1]:
                        neuron.error += neuron.error_func(next_neuron.error)
                else:
                    neuron.error = neuron.error_func(expected_out[i])

                # finally apply the cost function
                neuron.delta = neuron.error * neuron.cost(neuron.output)


def sigmoid(x):
    """
    @brief Represents the sigmoid activation function

    @param x: The value to perform the sigmoid on

    @return The sigmoid of x
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_cost(output):
    """
    @brief The derivative of the sigmoid function

    @param output: The output from the sigmoid function

    @return The slope
    """
    return output * (1 - output)


def hyperbolic_tangent(x):
    """
    @brief Represents the hyperbolic tangent activation function

    @param x: The value to perfrom the activation function on

    @return: The activation function applied to x
    """
    return (np.exp(x) - np.exp(-x)) / np.exp(x) + np.exp(-x)


def hyperbolic_cost(output):
    """
    @brief Represents the derivative of the hyperbolic tangent

    @param output: The output of the hyperbolic tangent

    @return: The slope
    """
    return 1 - np.square(np.tanh(output))


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
    # TODO add a bias "feature"
    network_sigmoid = Network(data, 3, False, sigmoid, sigmoid_cost)
    network_tan = Network(data, 3, False, hyperbolic_tangent, hyperbolic_cost)
    network_sigmoid.train_network(0.5, 100)
    network_tan.train_network(0.5, 100)