"""
    Machine Learning, University of Cincinnati
    Grace Gamstetter, Michael Gentile
    Assignment 4, Neural Networks
"""

import random
import numpy as np


class Data:
    def __init__(self):
        # Initialize class variables
        self.pairs = []
        self.error = []

    def generate_random_pairs(self, num_paris):
        self.pairs += [(random.randint(0, 10000), random.randint(0, 10000)) for _ in range(num_paris)]
        

    @staticmethod
    def pair_matches_concept(pair):
        """
        Return a bool based on whether they are in the positive class (true) or the negative class.
        Is what would be called an 'activation function'
        """
        return pair[0] + (2 * pair[1]) - 2 > 0


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

    def get_accuracy(self, predicted, truth):
        correct = 0
        for i in range(len(predicted)):
            if predicted[i] ==  truth[i]:
                correct = correct + 1
        return float(correct/float(len(predicted)))

    def get_weights(self):
        return self.get_weights

    def fit_with_update(self):
        # Find some sort of weight
        self.weights=[0.0 for i in range(len(data_obj.pairs) + 1)]
        # For each iteration
        for i in range(iterations):
            # For each example
            for j in range(len(data_obj.pairs)):
                interested_pair = data_obj.paris[j]
                predicted = self.activate(interested_pair)
                #check for misclassification
                if(y.iloc[j]==predicted):
                    pass
                else:
                    #calculate the error value
                    error=y.iloc[j]-predicted
                    #updation of threshold
                    self.weights[0]=self.weights[0] + self.learning_rate * error
                    #updation of associated self.weights acccording to Delta rule
                    for k in range(len(x)):
                        self.weights[k+1] = self.weights[k+1] + self.learning_rate * error * x[k]

    def fit_no_update(self):
        # Find some sort of weight
        self.weights=[0.0 for i in range(len(data_obj.pairs) + 1)]
        # For each iteration
        for i in range(iterations):
            # For each example
            for j in range(len(data_obj.pairs)):
                interested_pair = data_obj.paris[j]
                predicted = self.activate(interested_pair)
                #check for misclassification
                if(y.iloc[j]==predicted):
                    pass
                else:
                    #calculate the error value
                    error=y.iloc[j]-predicted
                    #updation of threshold
                    self.weights[0]=self.weights[0] + self.learning_rate * error

if __name__ == '__main__':
    data = Data()
    data.generate_random_pairs(100)
    delta_model = Delta(data)

