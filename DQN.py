import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import torch.optim as optim
from itertools import count
from simulator import Simulator
from reinforcement.core import ReplayMemory
from reinforcement.core import Transition
from reinforcement.core import ActionPicker
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 128
GAMMA = 0.999


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


environment=Simulator()
memory = ReplayMemory(10000)
actionpicker = ActionPicker(environment.nb_users*environment.nb_word)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        space = environment.action_space()[0] * environment.action_space()[1]
        
        self.layer1 = nn.Linear(space*2, space)
        #self.layer2 = nn.Linear(space * 2, space)
        #self.bn2 = nn.BatchNorm1d(space * 2)

        
    def forward(self, x):
        #sigm=torch.nn.Sigmoid()
        #x = sigm(self.layer1(x))
        x=self.layer1(x)
        return x


model = DQN()
optimizer = optim.RMSprop(model.parameters())



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))

    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch)
    state_action_values = state_action_values.gather(1, action_batch.view(-1,1))
    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 1000
for i_episode in range(num_episodes):
    print(i_episode)
    # Initialize the environment and state
    state = environment.reset()
    state = Tensor(state.reshape(1,2*environment.nb_users*environment.nb_word))
        
    for t in count():
        # Select and perform an action
        if True: #actionpicker.eps_decay():
            #Pick by the model
            bonus = actionpicker.ucb_bonus()
            temp = model(Variable(state, volatile=True).type(FloatTensor)).data
            action= (temp + FloatTensor(bonus)).max(1)[1].view(1, 1)
            
            #convert to tuples
            action=int(action.numpy()[0])
            actionpicker.ucb_action(action)
            action = (action//environment.nb_word,action % environment.nb_word)
        else:
            #Pick by random
            action=environment.uniform_sample()

        next_state, reward, done= environment.step(action)
        next_state = Tensor(next_state.reshape(1,2*environment.nb_users*environment.nb_word))
        #print(reward)
        reward = Tensor([reward])
        actiondo=LongTensor([action[0]*environment.nb_word+action[1]])


        # Store the transition in memory
        memory.push(state, actiondo, next_state, reward)

        # Move to the next state
        state = next_state
        print(state)
        print(model(
            Variable(state, volatile=True).type(FloatTensor)).data)
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            break

print('Complete')
#try out the policy

l_curve=environment.learning_curve

state = environment.reset()
state = Tensor(state.reshape(1,2*environment.nb_users*environment.nb_word))
for t in count():
    print(state)
    action = model(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    action=int(action.numpy()[0])
    action = (action//environment.nb_word,action % environment.nb_word)
    print(model(
            Variable(state, volatile=True).type(FloatTensor)).data)
    next_state, reward, done= environment.step(action)
    state = Tensor(next_state.reshape(1,2*environment.nb_users*environment.nb_word))
    if done:
        break

result=environment.gethistory()

#random agent
environment.reset()
for t in count():
    next_state, reward, done = environment.step(environment.uniform_sample())
    if done:
        break
randomresult=environment.gethistory()
#plot: random: blue, policy red
plt.plot(randomresult)
plt.plot(result,'r-')
plt.show()

plt.plot(l_curve)
plt.show()

