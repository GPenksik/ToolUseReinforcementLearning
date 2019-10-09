from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath("C:\\Users\\genia\\Documents\\Source\\Repos\\vs_drl_bootcamp1"))
sys.path.append(os.path.abspath("C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\Anaconda3_64\\Lib\\site-packages"))


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        dropout_p = 0.2
        n_hidden_1 = 128
        n_hidden_2 = 64
        n_fn_approx = 32
        self.fc1 = nn.Linear(4, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc_3_A = nn.Linear(n_hidden_2, n_fn_approx)
        self.fc_3_V = nn.Linear(n_hidden_2, n_fn_approx)
        self.V_fn = nn.Linear(n_fn_approx, 1)
        self.A_fn = nn.Linear(n_fn_approx, 4)
        

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        val_1 = F.relu(self.fc_3_V(x))
        val_2 = self.V_fn(val_1)



        adv_1 = F.relu(self.fc_3_A(x))
        adv_2 = self.A_fn(adv_1)
        
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        #if (len(x.shape) > 1):
        #    action = val_2 + adv_2 - adv_2.mean(0).unsqueeze(0).expand(4)
        #else:
        action = val_2 + adv_2 - adv_2.mean(1).unsqueeze(1).expand(input.size()[0],4)


        return action

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
