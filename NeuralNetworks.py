"""
TODO: File Header
"""

import random


class Data:
    def __init__(self):
        # Initialize class variables
        self.pairs = []
        self.error = []

    def generate_random_pairs(self, num_paris):
        self.pairs += [(random.randint(0, 10000), random.randint(0, 10000)) for _ in range(num_paris)]

    @staticmethod
    def pair_matches_concept(pair):
        return pair[0] + (2 * pair[1]) - 2 > 0


class Delta:
    def __init__(self, data_obj):
        self.data_obj = data_obj
        pass


if __name__ == '__main__':
    data = Data()
    data.generate_random_pairs(5)
    delta_model = Delta(data)
