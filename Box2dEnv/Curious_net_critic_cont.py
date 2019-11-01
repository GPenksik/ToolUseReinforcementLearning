from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_states, n_hidden=64):
        super(Net, self).__init__()
        n_hidden_1 = n_hidden
        n_hidden_2 = n_hidden
        self.fc1 = nn.Linear(n_states, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.out_baseline = nn.Linear(n_hidden_2, 1)
        # self.out_curious = nn.Linear(n_hidden_2, 1)
        # self.out_baseline.weight.data.mul(0.1)
        # self.out_curious.weight.data.mul(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_baseline = self.out_baseline(x)
        # x_curious = self.out_curious(x)

        return x_baseline  #, x_baseline

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
