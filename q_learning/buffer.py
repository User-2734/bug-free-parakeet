import random
from collections import deque


class ReplayBuffer(deque):
    def add(self, experience):
        self.append(experience)

    def sample(self, batch_size):
        return random.sample(self, batch_size)
