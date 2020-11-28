"""
    Machine Learning, University of Cincinnati
    Grace Gamstetter, Michael Gentile
    Assignment 4, Neural Networks
"""

import matplotlib.pyplot as plt
import random
import time
import numpy as np


class Data:
    def __init__(self):
        # Initialize class variables
        self.pairs = []
        self.truth_values = []

    def generate_random_pairs(self, num_paris):
        for i in range(num_paris):
            self.pairs.append((random.randint(-10000, 10000), random.randint(-10000, 10000)))
            self.truth_values.append(self.pair_matches_concept(self.pairs[i]))

    def pair_matches_concept(self, pair):
        """
        Return a bool based on whether they are in the positive class (true) or the negative class.
        Is what would be called an 'activation function'
        """
        return_value = pair[0] + (2 * pair[1]) - 2 > 0
        return return_value


class Delta:
    def __init__(self, data_obj, iterations, learning_rate):
        self.data_obj = data_obj
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = []
    
    def activate(self, interested_pair):
        activation = self.weights[0]
        activation += np.dot(self.weights[1:], interested_pair)
        if activation >= 0:
            return 1
        else:
            return 0

    def predict(self, test_set):
        end_predictions = []
        for i in range(len(test_set)):
            end_predictions.append(self.activate(test_set[i]))
        return end_predictions

    def get_accuracy(self):
        end_predictions = self.predict(self.data_obj.pairs)
        num_correct = [end_predictions[i] == self.data_obj.truth_values[i] for i in range(len(end_predictions))].count(True)
        return float(num_correct / len(self.data_obj.pairs))

    def get_weights(self):
        return self.weights

    def fit_with_update(self):
        # Find some sort of weight
        self.weights=[1.0 for i in range(len(self.data_obj.pairs[0]) + 1)]
        # For each iteration
        for i in range(self.iterations):
            # For each example
            for j in range(len(self.data_obj.pairs)):
                interested_pair = self.data_obj.pairs[j]
                predicted = self.activate(interested_pair)
                true_solution = self.data_obj.truth_values[j]
                # Check if something went wrong.
                if(true_solution == predicted):
                    # Classified correctly
                    pass
                else:
                    #Something went wrong. Find the error.
                    error = true_solution - predicted
                    # Change the weight.
                    self.weights[0] = self.weights[0] + self.learning_rate * error
                    
#                    Do the updating of all the weights due to the changes. 
                    for k in range(len(self.data_obj.pairs[j])):
                        self.weights[k+1] = self.weights[k+1] + self.learning_rate * error * self.data_obj.pairs[j][k]

    def fit_no_update(self):
        # Find some sort of weight
        self.weights=[1.0 for i in range(len(self.data_obj.pairs[0]) + 1)]
        # For each iteration
        for i in range(self.iterations):
            # For each example
            for j in range(len(self.data_obj.pairs)):
                interested_pair = self.data_obj.pairs[j]
                predicted = self.activate(interested_pair)
                true_solution = self.data_obj.truth_values[j]
                # Check if something went wrong.
                if(true_solution == predicted):
                    # Classified correctly
                    pass
                else:
                    #Something went wrong. Find the error.
                    error = true_solution - predicted
                    # Change the weight.
                    self.weights[0] = self.weights[0] + self.learning_rate * error
                    # Don't update.
            for k in range(len(self.data_obj.pairs[j])):
                self.weights[k+1] = self.weights[k+1] + self.learning_rate * error * self.data_obj.pairs[j][k]

    def fit_with_decay(self, decay_rate):
        self.weights=[1.0 for i in range(len(self.data_obj.pairs[0]) + 1)]
        initial_learning_rate = self.learning_rate
        # For each iteration
        for i in range(self.iterations):
            # For each example
            for j in range(len(self.data_obj.pairs)):
                interested_pair = self.data_obj.pairs[j]
                predicted = self.activate(interested_pair)
                true_solution = self.data_obj.truth_values[j]
                # Check if something went wrong.
                if(true_solution == predicted):
                    # Classified correctly
                    pass
                else:
                    #Something went wrong. Find the error.
                    error = true_solution - predicted
                    # Change the weight.
                    self.weights[0] = self.weights[0] + self.learning_rate * error
                    # Do the updating of all the weights due to the changes. 
                    for k in range(len(self.data_obj.pairs[j])):
                        self.weights[k+1] = self.weights[k+1] + self.learning_rate * error * self.data_obj.pairs[j][k]
            self.learning_rate = (decay_rate ** (self.iterations + 1)) * initial_learning_rate

    def fit_with_adaptive(self, threshold, learning_rate_decrease_rate, learning_rate_increase_rate):
        self.weights=[0.0 for i in range(len(self.data_obj.pairs[0]) + 1)]
        initial_learning_rate = self.learning_rate
        incorrect = 0
        previous_error = 0.0
        # For each iteration
        for i in range(self.iterations):
            # For each example
            for j in range(len(self.data_obj.pairs)):
                interested_pair = self.data_obj.pairs[j]
                predicted = self.activate(interested_pair)
                true_solution = self.data_obj.truth_values[j]
                # Check if something went wrong.
                if(true_solution == predicted):
                    # Classified correctly
                    pass
                else:
                    incorrect = incorrect + 1
                    #Something went wrong. Find the error.
                    error = true_solution - predicted
                    # Change the weight.
                    self.weights[0] = self.weights[0] + self.learning_rate * error
                    # Do the updating of all the weights due to the changes. 
                    for k in range(len(self.data_obj.pairs[j])):
                        self.weights[k+1] = self.weights[k+1] + self.learning_rate * error * self.data_obj.pairs[j][k]
            if i > 0:
                # Don't do it on the first pass. 
                if incorrect / float(len(self.data_obj.pairs)) > threshold:
                    # Discard the previous weights.
                    self.weights=[0.0 for i in range(len(self.data_obj.pairs[0]) + 1)]
                    self.learning_rate = self.learning_rate * learning_rate_decrease_rate
                else:
                    self.learning_rate = self.learning_rate * learning_rate_increase_rate
            previous_error = incorrect / float(len(self.data_obj.pairs))




def show_plot(iterations, accuracies, type, learning_rate):
    plt.scatter(iterations, accuracies)
    plt.xlabel("Num Iterations")
    plt.ylabel("Accuracy")
    title = type + " Accuracy for " + str(learning_rate)
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    data = Data()
    data.generate_random_pairs(100)
    # Get a set of learning rates to test and explain.
    learning_rates = [0.001, 0.01, 0.1, 0.2, 0.3]
    iterations = [5, 10, 50, 100]
    types = ["Standard Delta", "Incremental Delta", "Decaying Rates", "Adaptive Rates"]
    for k in range(len(types) - 2):
        for i in range(len(learning_rates)):
            accuracy = []
            for j in range(len(iterations)):
                if k == 0:
                    delta_model = Delta(data, iterations[j], learning_rates[i])
                    start_time = time.time()
                    delta_model.fit_no_update()
                    print delta_model.get_weights()
                    print "--- " + str(time.time() - start_time) + " seconds ---"
                    accuracy.append(delta_model.get_accuracy())
                if k == 1:
                    delta_model = Delta(data, iterations[j], learning_rates[i])
                    start_time = time.time()
                    delta_model.fit_with_update()
                    print delta_model.get_weights()
                    print "--- " + str(time.time() - start_time) + " seconds ---"
                    accuracy.append(delta_model.get_accuracy())
            show_plot(iterations, accuracy, types[k], learning_rates[i])

    start_time = time.time()
    delta_model = Delta(data, 50, 0.2)
    delta_model.fit_with_decay(0.8)
    print delta_model.get_weights()
    
    delta_model = Delta(data, 50, 0.2)
    delta_model.fit_with_adaptive(.9, .95, 1.05)
    print delta_model.get_weights()
    print "--- " + str(time.time() - start_time) + " seconds ---"
