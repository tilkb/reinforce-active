import torch
import torch.nn as nn
from simulator import Simulator
from itertools import count
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from reinforcement.metrics import Metrics
import numpy as np
from torch.distributions import Categorical


GAMMA = 0.995

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


environment=Simulator()


class PolicyNetwork(nn.Module):

    def __init__(self):
        super(PolicyNetwork, self).__init__()
        space = environment.action_space()[0] * environment.action_space()[1]
        self.smax= nn.Softmax()    
        self.layer2 = nn.Linear(space * 2, space)

        self.saved_log_probs = []
        self.rewards = []

        
    def forward(self, x):
        #sigm=torch.nn.Sigmoid()
        #x = sigm(self.layer1(x))
        x = self.smax(self.layer2(x))
        return x


policy_network = PolicyNetwork()
if use_cuda:
    policy_network.cuda()
optimizer = optim.Adam(policy_network.parameters())


def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy_network(Variable(state))
    m = Categorical(probs)
    action = m.sample()
    policy_network.saved_log_probs.append(m.log_prob(action))
    action2 = int(action.cpu().data.numpy()[0])
    action2 = (action2//environment.nb_word,action2 % environment.nb_word)
    return action2



def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy_network.rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for log_prob, reward in zip(policy_network.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy_network.rewards[:]
    del policy_network.saved_log_probs[:]









num_episodes = 10000
for i_episode in range(num_episodes):
    print(i_episode)
    # Initialize the environment and state
    state = environment.reset() 
    state = state.reshape(1,2*environment.nb_users*environment.nb_word) 
    for t in count():
        action = select_action(state)

        next_state, reward, done= environment.step(action)
        state = next_state.reshape(1,2 * environment.nb_users * environment.nb_word)
        
        policy_network.rewards.append(reward)
        if done:
            break
 
    finish_episode()
    

l_curve=environment.learning_curve
plt.plot(l_curve)
plt.show()

def policy(state):
    probs = policy_network(
            Variable(state, volatile=True).type(FloatTensor))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    action=int(action.cpu().data.numpy()[0])
    action = (action//environment.nb_word,action % environment.nb_word)
    return action

metrics = Metrics(environment)
inc = metrics.compare_policy([("PolicyGradient",policy)],10,True)

print(inc)

