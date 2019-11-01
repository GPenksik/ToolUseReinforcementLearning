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

    def __init__(self, n_states, n_hidden=64):
        super(Net, self).__init__()
        n_hidden_1 = n_hidden
        n_hidden_2 = n_hidden
        self.fc1 = nn.Linear(n_states, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.out = nn.Linear(n_hidden_2, 1)
        self.out.bias.data = torch.tensor(0).float()
        # torch.nn.init.kaiming_normal_(self.out.weight.data)
        # torch.nn.init.kaiming_normal_(self.fc1.weight.data)
        # torch.nn.init.kaiming_normal_(self.fc2.weight.data)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
