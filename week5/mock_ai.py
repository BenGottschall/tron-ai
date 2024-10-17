import random

class MockAI:
    def __init__(self):
        self.directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    def get_direction(self, *args):
        return random.choice(self.directions)