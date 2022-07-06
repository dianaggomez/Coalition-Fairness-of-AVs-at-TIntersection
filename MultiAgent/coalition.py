import numpy as np
from itertools import product
import time
import random

class Coalition():
    def __init__(self, coalition_num=0, num_of_vehicles=6, policy ='SR'):
        self.coalition_num = coalition_num
        self.num_of_vehicles = num_of_vehicles
        self.policy = policy

        self.Q = 0 # [0, 1]

        self.time_steps = 0

        self.t_SR = 0
        self.t_pi = 0
        self.agents_t_SR = 0
        self.agents_t_pi = 0

    def choose_action(self, s, EPSILON=0.1):
        # Possible Actions
        # 0: stop, 1: go

        GAMMA = 0.95
        ALPHA = 0.1

        if self.policy == 'greedy': 
            explore = np.random.choice([0, 1], 1, p=[1-EPSILON, EPSILON])
            if explore:
                return np.random.choice(4)
            else:
                q_star = np.argmax(self.Q[s, :])
                idx_max = np.where(self.Q[s, :]==self.Q[s, :][q_star])[0]
                # break ties
                if idx_max.shape[0] > 1:
                    return np.random.choice(idx_max)
                else:
                    return np.argmax(self.Q[s, :])
        else:
            return 1
    