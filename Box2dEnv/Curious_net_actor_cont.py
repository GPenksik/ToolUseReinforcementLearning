from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath("C:\\Users\\genia\\Documents\\Source\\Repos\\vs_drl_bootcamp1"))
sys.path.append(os.path.abspath("C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\Anaconda3_64\\Lib\\site-packages"))


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, n_states, n_actions, n_hidden=64):
        super(Net, self).__init__()

        n_hidden_1 = n_hidden
        n_hidden_2 = n_hidden
        #n_hidden_3 = 64
        self.fc1 = nn.Linear(n_states, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        #self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.outmu1 = nn.Linear(n_hidden_2, n_actions)
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc1.weight.data)
        torch.nn.init.kaiming_uniform_(self.outmu1.weight.data)
        self.outmu1.weight.data.mul_(0.01)

        self.logstd = nn.Parameter(torch.add(torch.zeros(n_actions), 0))

    def forward(self, input):
        x = F.tanh(self.fc1(input))
        x = F.tanh(self.fc2(x))
        #x = F.tanh(self.fc3(x))
        mu = F.tanh(self.outmu1(x))*1.1
        logstd = self.logstd.expand_as(mu)
        std = torch.exp(logstd)

        return mu, std

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
