import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import torch.optim as optim
from itertools import count
from simulator import Simulator
from simulator import ParalellSyncronSimulator
from reinforcement.PriorityReplayMemory import PriorityReplayMemory
from reinforcement.core import ReplayMemory
from reinforcement.core import Transition
from reinforcement.core import ActionPicker
import matplotlib.pyplot as plt
import numpy as np
from reinforcement.metrics import Metrics

paralell_simulator = 1
BATCH_SIZE = 128
GAMMA = 0.9
PRIORITY_REPLAY = False



# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

par_environment = ParalellSyncronSimulator(paralell_simulator)
environment=Simulator()
if PRIORITY_REPLAY:
    memory = PriorityReplayMemory(10000, BATCH_SIZE, 1.1)
else:
    memory = ReplayMemory(10000)
actionpicker = ActionPicker(environment.nb_users*environment.nb_word)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        space = environment.action_space()[0] * environment.action_space()[1]
        
        self.layer1 = nn.Linear(space*2, space)
        #self.layer2 = nn.Linear(space, space)
        #self.bn2 = nn.BatchNorm1d(space * 2)

        
    def forward(self, x):
        #sigm=torch.nn.Sigmoid()
        #x = sigm(self.layer1(x))
        x=self.layer1(x)
        return x

model1 = DQN()
model2 = DQN()
if use_cuda:
    model1.cuda()
    model2.cuda()
optimizer1 = optim.RMSprop(model1.parameters())
optimizer2 = optim.RMSprop(model2.parameters())


def optimize_model():
    if PRIORITY_REPLAY:
        batch = memory.select(1.0)
        print(batch)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
    else:
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

    state_action_values = model1(state_batch)
    state_action_values = state_action_values.gather(1, action_batch.view(-1,1))
    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model2(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer1.zero_grad()
    loss.backward()
    for param in model1.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer1.step()


def main():
    global model1
    global model2
    global optimizer1
    global optimizer2

    num_episodes = 3000

    for i_episode in range(num_episodes):
        print(i_episode)
        # Initialize the environment and state

        state = np.array(par_environment.reset())
        state = Tensor(state.reshape(paralell_simulator,2*environment.nb_users*environment.nb_word))

        
        for t in count():
            # Select and perform an action
            if True: #actionpicker.eps_decay():
                #Pick by the model
                bonus = np.repeat(actionpicker.ucb_bonus(),paralell_simulator,axis=0).reshape((paralell_simulator, environment.nb_users*environment.nb_word))
                temp = model1(Variable(state, volatile=True).type(FloatTensor)).data
                action= (temp + FloatTensor(bonus)).max(1)[1].view(paralell_simulator, 1)
            
                #convert to tuples
                actionlist=[]
                action=action.cpu().numpy()
                for act in action:
                    tmp_act = int(act)
                    actionpicker.ucb_action(tmp_act)
                    actionlist.append((tmp_act//environment.nb_word,tmp_act % environment.nb_word))
            else:
                #Pick by random
                action=environment.uniform_sample()
            next_state, reward, done = par_environment.step(actionlist)
            next_state = Tensor(next_state.reshape(paralell_simulator,2*environment.nb_users*environment.nb_word))
            
            for idx, item in enumerate(actionlist):
                actiondo=LongTensor([item[0]*environment.nb_word+item[1]])
                # Store the transition in memory
                memory.push(state[idx].view(1,-1), actiondo, next_state[idx].view(1,-1), Tensor([reward[idx]]))
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the target network)
            optimize_model()
            if all(done):
                break
        #swap models after every episode
        optimizer1, optimizer2 = optimizer2, optimizer1
        model1, model2 = model2, model1

    print('Complete')
    #try out the policy

    l_curve=environment.learning_curve
    plt.plot(l_curve)
    plt.show()

    metrics = Metrics(environment)
    inc = metrics.compare_policy([("DDQN",DQN_policy)],30,False)
    torch.save(model1,"saved_policy/DDQN.pkl")
    print(inc)

def DQN_policy(state):
    action = model1(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    action=int(action.cpu().numpy()[0])
    action = (action//environment.nb_word,action % environment.nb_word)
    return action


if __name__ == "__main__":
    main()



