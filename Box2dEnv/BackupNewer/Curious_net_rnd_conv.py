from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath("C:\\Users\\genia\\Documents\\Source\\Repos\\vs_drl_bootcamp1"))
sys.path.append(os.path.abspath("C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\Anaconda3_64\\Lib\\site-packages"))


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    n_hidden_1 = 64
    def __init__(self, n_states, n_channel=16, n_hidden=64):
        super(Net, self).__init__()
        n_hidden_1 = n_hidden
        n_hidden_2 = n_hidden
        self.n_hidden_1 = n_hidden_1

        self.conv1 = nn.Conv1d(1, n_channel, 5, stride=1)
        self.conv2 = nn.Conv1d(n_channel, n_hidden_1, 5, stride=1)
        # self.fc4 = nn.Linear(n_hidden_1*(n_states-4), n_hidden_2)
        self.out = nn.Linear(n_hidden_1 * (n_states-8), 1)
        #self.out.bias.data = torch.add(self.out.bias.data, -0.1)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.n_hidden_1*x.shape[2])
        # x = F.relu(self.fc4(x))
        x = self.out(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features