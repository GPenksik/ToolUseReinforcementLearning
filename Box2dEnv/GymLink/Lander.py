import sys
import os
sys.path.append(os.path.abspath("C:\\Users\\genia\\source\\repos\\Box2dEnv"))

import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import lander_net_DuelDDQN as tor
import collections
from collections import deque
env = gym.make('EnvTest-v2')

load = False

np.set_printoptions(precision = 3)
torch.set_printoptions(precision = 3)

# Initialise network and hyper params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type('torch.FloatTensor')
net = tor.Net()#.to(device)
target_net = tor.Net()
target_net.eval()
#criterion = nn.SmoothL1Loss()
criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.005, weight_decay=0.01)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
gamma = torch.tensor(0.99)

BATCH_SIZE = 32
REPLAY_SIZE = 100000

TARGET_BURNIN = 2001
TARGET_UPDATE = 100

EPSILON_START = 0.8
EPSILON_END = 0.05

EPSILON_DECAY_LENGTH = 150000

N_ACTIONS = 4
N_STATES = 4

TAU_START = 0.05
TAU_END = 0.05
TAU_DECAY_LENGTH = 75000

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

if load == True:
    net = torch.load("C:\\Users\\genia\\Documents\\Source\\Repos\\vs_drl_bootcamp1\\savedNetLander-episode300.pth")
    #net.load_state_dict(net_dict)
    target_net.load_state_dict(net.state_dict())
else:
    #net.apply(init_normal)
    target_net.load_state_dict(net.state_dict())

D_state = torch.empty(REPLAY_SIZE,N_STATES)
D_state_p = torch.empty(REPLAY_SIZE,N_STATES)
D_reward = torch.empty(REPLAY_SIZE,1)
D_action = torch.empty(REPLAY_SIZE,1)
D_done = torch.empty(REPLAY_SIZE,1)


D_state_dq = deque(maxlen=REPLAY_SIZE)

# Initialise and render environment
cur_state = env.reset()
i_global = 0

env.render()


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def push_to_tensor(tensor, x, pos):
    #tensor[1:] = tensor[:-1]
    tensor[pos] = x
    return tensor

# Re initialise episode
def train(i, episodes, train=False):
    cur_state = env.reset()
    history_capacity = 0
    history_i = 0
    epsilon_decay = (EPSILON_START-EPSILON_END)/EPSILON_DECAY_LENGTH
    tau_decay = (TAU_START - TAU_END)/TAU_DECAY_LENGTH
    training_started = False
    epsilon = EPSILON_START
    tau = TAU_START
    cul_reward = deque(maxlen=15)
    
    for episode in range(episodes):
        ret = 0.
        done = False
        cur_state = env.reset()
        cul_loss = 0
        cul_q = 0

        episode_i = 1
        while not done:
            with torch.no_grad():
                q_current = target_net(torch.tensor(cur_state).float().unsqueeze(0))

            random_epsilon = np.random.random()
            if random_epsilon < epsilon:
                if np.random.random() < 0.0:
                    action = 0
                else:
                    action = env.action_space.sample() # sample an action randomly
                q_action = q_current.data[0][action].item()
            else:
                q_action = torch.max(q_current).item()
                action = torch.argmax(q_current).item()
            
            next_state, reward, done, [] = env.step(action)

            reward_clipped = reward#/300
            #if reward > 1:
            #    reward_clipped = 1
            #elif reward < -1:
            #    reward_clipped = -1

            #if episode_i == 1001 and done == True:
            #    done = False
                

            ret += reward

            cul_q += q_action

            if episode_i > 10000:
                done = True

            if done == True:
                inv_done = 0
            else:
                inv_done = 1


                #print("Pre-filling reply buffer", history_capacity)
            if history_capacity < REPLAY_SIZE:
                history_capacity += 1

            if history_i < REPLAY_SIZE:
                history_i += 1
            else:
                history_i = 0

            push_to_tensor(D_state, torch.tensor(cur_state), history_i-1)
            push_to_tensor(D_state_p, torch.tensor(next_state), history_i-1)
            push_to_tensor(D_reward, torch.tensor(reward_clipped), history_i-1)
            push_to_tensor(D_action, torch.tensor(action), history_i-1)
            push_to_tensor(D_done, torch.tensor(inv_done), history_i-1)

            if history_capacity > BATCH_SIZE and train==True:
                if training_started == False:
                    print("Training now started. Saving model")
                    torch.save(net.state_dict(),"savedNetLander-episode"+str(0)+".pth")
                    training_started = True

                batch_indexes = torch.tensor(np.random.permutation(history_capacity)[0:BATCH_SIZE]).type(torch.LongTensor)#.cuda() ##
                #batch_indexes_dq = batch_indexes.cpu().data.numpy()
                #dq_length = len(batch_indexes_dq)
                #batch_input_dq = torch.empty(dq_length,4) 
                #for dq_i in range(dq_length):
                #    batch_input_dq[dq_i] = D_state_dq[batch_indexes[dq_i]]

                batch_action = torch.index_select(D_action, 0, batch_indexes).type(torch.LongTensor)#.cuda() ##

                batch_input = torch.index_select(D_state, 0, batch_indexes)
                batch_input_p = torch.index_select(D_state_p, 0, batch_indexes)
                batch_reward = torch.index_select(D_reward, 0, batch_indexes)    
                batch_done_gamma = torch.index_select(D_done, 0, batch_indexes) * gamma

                #for i_check in range(history_capacity):
                #    if (D_state[i_check] == D_state_dq[i_check]).all().item() == False:
                #        print("WOOOAH, Stop right there")

                #if (batch_input == batch_input_dq).all() == False:
                #    print("SHITTY INPUT DETECTED!!")

                with torch.no_grad():
                    out_yi_Q = net(batch_input_p)
                    out_yi_Q_max_a = torch.argmax(out_yi_Q,1).type(torch.LongTensor)#.cuda()
                    out_yi = target_net(batch_input_p)
                    out_yi_max_raw = torch.gather(out_yi, 1, out_yi_Q_max_a.unsqueeze(1))
                    out_yi_max = batch_reward + torch.mul(out_yi_max_raw,batch_done_gamma)

                out_fwd_raw = net(batch_input)

                out_fwd = torch.gather(out_fwd_raw, 1, batch_action)

                loss = criterion(out_fwd, out_yi_max)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(net.parameters(),0.1)
                optimizer.step()

                if i > TARGET_BURNIN:
                    soft_update(net, target_net, tau)

                cul_loss += loss.item()
                episode_i += 1
                i += 1

                if np.any(np.isnan(net.fc1.weight.cpu().data)):
                    print("ITS A NAN! STOP")

                if i < TARGET_BURNIN:
                    target_net.load_state_dict(net.state_dict())
                    #print("Updating target burnin")

                if epsilon > EPSILON_END:
                    epsilon -= epsilon_decay

                if tau > TAU_END and i > TARGET_BURNIN:
                    tau -= tau_decay

                if i < TARGET_BURNIN and i % TARGET_UPDATE == 0:
                    target_net.load_state_dict(net.state_dict())
                    #print("Saving target")
                    
                if episode % 100 == 0 and done == True and i > 300:
                    #target_net.load_state_dict(net.state_dict())
                    #print("Saving net")
                    torch.save(net,"savedNetLander-episode"+str(episode)+".pth")
                    #torch.save(target_net.state_dict(), "savedTNet-episode"+str(episode)+".pt")

            cur_state = next_state
            
            if i % 5 == 0 and i > 20000:
                env.render()
            #time.sleep(.01)

            #if episode % 200 == 0 and epsilon > 0.051 and episode_i == 1:
            #    epsilon = epsilon - 0.025
            #    print("Epsilon is now ", epsilon)
            #    break # for the purpose of this visualization, let's only run for 1500 steps
            #    # also note the GUI won't close automatically

        
        cul_reward.append(ret)
        
        print("%4d, %6d, %4d, %8.2f, %5.1f, %10.4f, %6.2f, %3.2f, %5.3f" % (episode, i, episode_i, ret, np.mean(cul_reward), cul_loss/episode_i, cul_q/episode_i, epsilon, tau))
    return i

i_global = train(i_global, 5000, train=True)
print(i_global)
env.close()