import sys
import os
import gym
import torch
import torch.nn as nn
import numpy as np

import Curious_net_rnd_conv as Net_rnd
from collections import deque
from torch.distributions import Normal

sys.path.append(os.path.abspath("C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv"))

load = False

N_STATES = 2
N_CURIOUS_STATES = 100
N_ACTIONS = 1
np.set_printoptions(precision=3, linewidth=200, floatmode='fixed', suppress=True)
torch.set_printoptions(precision=3)

# Initialise network and hyper params
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type('torch.FloatTensor')
N_HIDDEN = 64
N_HIDDEN_RND = 16
N_CHANNELS_RND = 16
ac_net_rnd = Net_rnd.Net(N_CURIOUS_STATES, N_CHANNELS_RND, N_HIDDEN_RND)
ac_net_pred = Net_rnd.Net(N_CURIOUS_STATES, N_CHANNELS_RND*2, N_HIDDEN_RND)

criterion_val = nn.MSELoss()

optimizer_rnd = torch.optim.SGD(ac_net_pred.parameters(), lr=0.001, momentum=0.0, nesterov=False)

return_time = 1

N_STEPS = 100
# N_STEPS = 500
N_TRAJECTORIES = 12
K_epochs = 5
B_epochs = 5
R_epochs = 1
N_MINI_BATCH = 256
avg_reward = deque(maxlen=100)
avg_curious_reward = deque(maxlen=100)
avg_STD = deque()
avg_critic_loss = deque()
avg_reward_STD = deque()
avg_value_STD = deque()

p1 = np.random.normal(0, 5, (N_CURIOUS_STATES, 1))
p2 = np.random.normal(0, 5, (N_CURIOUS_STATES, 1))


def clip_min_grad_value_(parameters, clip_value):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = torch.tensor(float(clip_value))
    for p in filter(lambda p: p.grad is not None, parameters):
        # p.grad.data.clamp_(min=-clip_value, max=clip_value)
        # p.grad.data = torch.min(torch.max(p.grad.data, clip_value), -clip_value)
        p.grad.data = torch.gt(torch.abs(p.grad.data), clip_value).float()*p.grad.data


def get_curious_state(state):
    return 1


# noinspection PyCallingNonCallable
def train(episodes):
    first_batch = True
    episode_i = 0
    total_i = 0
    curious_reward_std = 0.2
    while episode_i < episodes:  # START MAIN LOOP
        cur_state_q = []

        if episode_i < 3:
            std1 = 0.3
            std2 = 0.3
            mean1 = 1
            mean2 = 3
        else:
            std1 = 0.8
            std2 = 0.3
            mean1 = 1

        cur_state_t = torch.tensor(cur_state_q).float()
        state_t = np.random.normal(0, std1, (4000, 1))
        state_t = state_t + mean1
        state_t_new = np.zeros((len(state_t), N_CURIOUS_STATES))
        state_t2 = np.random.normal(0, std2, (4000, 1))
        state_t2 = state_t2 + mean2
        state_t2_new = np.zeros((len(state_t2), N_CURIOUS_STATES))
        for x, p1x, p2x, p1y, p2y in zip(range(N_CURIOUS_STATES), p1, p2, reversed(p1), reversed(p2)):
            state_t_new[:, x] = np.squeeze(p1x*np.cos(p2x*state_t) + p1y * np.sin(p2y*state_t))
            state_t2_new[:, x] = np.squeeze(p1x*np.cos(p2x*state_t2) + p1y * np.sin(p2y*state_t2))


        # curious_state_t = torch.tensor(np.zeros(10000,)).unsqueeze(1).float()
        # curious_state_t = torch.mul(torch.mul(torch.rand(10000), 0), 5).unsqueeze(1).float()
        #state_t = torch.add(state_t, 3.5)
        # state_t = torch.tensor(state_t_new)
        curious_state_t = torch.tensor(state_t_new).float()
        curious_state_t2 = torch.tensor(state_t2_new).float()

        # flip = -1
        # for n_i in range(5):
        #     curious_state_t = torch.cat((curious_state_t, flip*torch.add(state_t, -(n_i+1))), 1)
        #     flip = -flip

        # curious_state_t = torch.div(torch.mul(curious_state_t, 30).int().float(), 30)

        if first_batch:
            test_state2 = np.arange(start=0, stop=5, step=0.001)
            test_state = np.zeros((len(test_state2), N_CURIOUS_STATES))
            for x, p1x, p2x, p1y, p2y in zip(range(N_CURIOUS_STATES), p1, p2, reversed(p1), reversed(p2)):
                test_state[:, x] = np.squeeze(p1x * np.cos(p2x * test_state2) + p1y * np.sin(p2y * test_state2))
            test_state = torch.tensor(test_state).float()

        with torch.no_grad():
            test_rnd_val = ac_net_rnd(test_state.unsqueeze(1))
            test_pred_val = ac_net_pred(test_state.unsqueeze(1))
            reward_rnd_val = ac_net_rnd(curious_state_t.unsqueeze(1))
            reward_pred_val = ac_net_pred(curious_state_t.unsqueeze(1))
            reward_rnd_val2 = ac_net_rnd(curious_state_t2.unsqueeze(1))
            reward_pred_val2 = ac_net_pred(curious_state_t2.unsqueeze(1))

        if first_batch:
            test_rewards_norm = torch.abs(test_rnd_val - test_pred_val)  # - test_pred_val)
            first_batch = False
            test_rewards = torch.pow((reward_rnd_val - reward_pred_val), 2)  # / test_rewards_norm
            test_rewards2 = torch.pow((reward_rnd_val2 - reward_pred_val2), 2)  # / test_rewards_norm

            print(torch.mean(test_rewards).data.numpy(), torch.mean(test_rewards2).data.numpy())



        # START CURIOUS OPTIMIZE
        batch_length = len(curious_state_t)
        if first_batch:
            n_mini_batch = batch_length
            # batch_state_t_const = batch_state_t
            cur_state_t_const = cur_state_t



        n_mini_batch = len(curious_state_t)
        # batch_length = n_mini_batch
        avg_curious_loss = []
        for epoch in range(R_epochs):
            # Get random permutation of indexes
            indexes = torch.tensor(np.random.permutation(batch_length)).type(torch.LongTensor)
            # indexes = torch.arange(batch_length).type(torch.LongTensor)
            n_batch = 0
            batch_start = 0
            batch_end = 0
            # Loop over permutation
            # avg_curious_loss = []
            while batch_end < batch_length:
                # Get batch indexes
                batch_end = batch_start + N_MINI_BATCH
                if batch_end > batch_length:
                    batch_end = batch_length

                batch_idx = indexes[batch_start:batch_end]

                # Gather data from saved tensors
                batch_state_t = torch.index_select(curious_state_t, 0, batch_idx).float()
                batch_start = batch_end
                n_batch += 1

                with torch.no_grad():
                    rnd_val = ac_net_rnd(batch_state_t.unsqueeze(1))
                pred_val = ac_net_pred(batch_state_t.unsqueeze(1))
                # Calculate loss compared with reward and optimize
                pred_loss_batch_curious = criterion_val(pred_val, rnd_val)
                optimizer_rnd.zero_grad()
                pred_loss_batch_curious.backward()
                nn.utils.clip_grad_norm(ac_net_pred.parameters(), 0.5)
                # nn.utils.clip_grad_value_(ac_net_pred.parameters(), 0.1)
                # clip_min_grad_value_(ac_net_pred.parameters(), 0.2)
                #nn.utils.clip_grad_norm(ac_net_pred.parameters(), 5)

                optimizer_rnd.step()
                avg_curious_loss.append(pred_loss_batch_curious.item())

            #print((pred_loss_batch_curious.data.numpy()), " ", end="")
            # print("")
            # print(epoch)
        #print("")

        # End curious optimization

        with torch.no_grad():
            #test_rnd_val = ac_net_rnd(test_state.unsqueeze(1))
            #test_pred_val = ac_net_pred(test_state.unsqueeze(1))
            test_rnd_val = ac_net_rnd(curious_state_t.unsqueeze(1))
            test_pred_val = ac_net_pred(curious_state_t.unsqueeze(1))
            test_rnd_val2 = ac_net_rnd(curious_state_t2.unsqueeze(1))
            test_pred_val2 = ac_net_pred(curious_state_t2.unsqueeze(1))
        # chunks = 20
        # rewards_mean = []
        # for rewards in torch.chunk(test_rewards, chunks):
        #     # print(torch.mean(rewards).data.numpy(), end="")
        #     rewards_mean.append(torch.mean(rewards).data.numpy())
        # # test_rewards = test_pred_val
        # rewards_mean = np.asarray(rewards_mean)
        test_rewards = torch.pow((test_rnd_val - test_pred_val), 2)  # / test_rewards_norm
        test_rewards2 = torch.pow((test_rnd_val2 - test_pred_val2), 2)  # / test_rewards_norm

        print(torch.mean(test_rewards).data.numpy(), torch.mean(test_rewards2).data.numpy(), np.mean(avg_curious_loss))
        # print(test_rewards.transpose(0, 1).data.numpy(), np.mean(avg_curious_loss))


        episode_i += 1
        # if episode_i % return_time == 0:
        #     print("%4d, %6.0d, %6.2f, %6.2f, %6.2f"
        #           % (episode_i, total_i, np.mean(avg_reward_batch), np.mean(avg_reward), np.mean(curious_reward_q)))
        # END UPDATING ACTOR

    return episode_i
    # END MAIN LOOP




i_global = train(500)
