from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from reinforcement.metrics import Metrics
from simulator import Simulator
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from itertools import count
from tqdm import tqdm

GAMMA = 0.99
environment=Simulator()


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        hidden = 256
        space = environment.action_space()[0] * environment.action_space()[1]
        self.affine1 = nn.Linear(space*2, hidden)
        self.action_head = nn.Linear(hidden, space)
        self.value_head = nn.Linear(hidden, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x= F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-2)


def select_action(state):
    state = torch.from_numpy(state).float().view(1,-1)
    probs, state_value = model(Variable(state))
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.data[0]


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + GAMMA* R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0] #calculate the adventage
        policy_losses.append(-log_prob * Variable(reward))
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    for i_episode in tqdm(range(1000)):
        state = environment.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            action = (action//environment.nb_word,action % environment.nb_word)
            state, reward, done = environment.step(action)
            model.rewards.append(reward)
            if done:
                break
        #print(i_episode)
        finish_episode()
    
    metrics = Metrics(environment)
    inc = metrics.compare_policy([("ActorCritic",AC_policy)],30,False)
    torch.save(model,"saved_policy/ActorCritic.pkl")
    print(inc)


def AC_policy(state):
    state = state.view(1,-1)
    probs, state_value = model(Variable(state))
    m = Categorical(probs)
    action = m.sample().data[0]
    action = (action//environment.nb_word,action % environment.nb_word)
    return action


if __name__ == '__main__':
    main()