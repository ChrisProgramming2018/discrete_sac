from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical



class QNetwork(nn.Module):
    def __init__(self, state_shape, n_actions, seed=0, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_shape, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, n_actions)
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(PolicyNetwork, self).__init__()
        self.n_actions = n_actions

        self.fc = nn.Linear(in_features=state_shape, out_features=512)
        self.logits = nn.Linear(in_features=512, out_features=self.n_actions)


    def forward(self, x):
        x = F.relu(self.fc(x))
        logits = self.logits(x)
        probs = F.softmax(logits, -1)
        z = probs == 0.0
        z = z.float() * 1e-8
        return Categorical(probs), probs + z
