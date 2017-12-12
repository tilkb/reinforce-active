import pickle
from simulator import Simulator
import torch
import torch.nn as nn
from reinforcement.metrics import Metrics
from torch.autograd import Variable
from PolicyGradient import PolicyNetwork
from PolicyGradient import policy
from DQN import DQN
from DQN import DQN_policy

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

environment=Simulator()


policy_network = torch.load("saved_policy/REINFORCE.pkl")
model = torch.load("saved_policy/DQN.pkl")

metrics = Metrics(environment)
inc = metrics.compare_policy([("PolicyGradient",policy),("DQN",DQN_policy)], 20, False)

print(inc)

