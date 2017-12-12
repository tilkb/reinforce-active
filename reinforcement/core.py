import random
import math
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class ActionPicker(object):
    def __init__(self, nb_action = 0):
        self.steps_done = 0
        #Constants
        self.EPS_START = 0.95
        self.EPS_END = 0.15
        self.EPS_DECAY = 400
        #boltzman

        #UCB
        self.time = nb_action
        self.used_actions = np.ones(nb_action)

    """Choose between explore and expoitation
    TRUE=Pick the best action
    FALSE=Pick a random action"""
    def eps_decay(self):
        #eps decay exploration function
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return True
        else:
            return False


    def boltzman_action_picker(self,vector):
        ex = np.exp(vector - np.max(vector))
        prob = ex / np.sum(ex)
        rand = random.random.choice(len(vector),1,p=prob)
        return rand


    def ucb_action(self,action):
        self.time = self.time + 1
        self.used_actions[action] = self.used_actions[action] + 1


    #return the bonus for the actions
    def ucb_bonus(self):
        bonus = np.sqrt(2*math.log(self.time)/self.used_actions)
        return bonus



