import sys
import os
sys.path.append(os.path.abspath("C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv"))

import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import Lander_net_actor_cont as AC_Net_Actor
import Lander_net_critic_cont as AC_Net_Critic
import collections
from collections import deque
env = gym.make('EnvTestContinuousR-v2')
from torch.distributions import Normal

load = False

N_STATES = 5
N_ACTIONS = 3
np.set_printoptions(precision = 3)
torch.set_printoptions(precision = 3)

# Initialise network and hyper params
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type('torch.FloatTensor')
ac_net_critic = AC_Net_Critic.Net(N_STATES)
ac_net_actor = AC_Net_Actor.Net(N_STATES, N_ACTIONS)

criterion_val = nn.MSELoss()
#optimizer_c = torch.optim.Adam(ac_net_critic.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
#optimizer_a = torch.optim.Adam(ac_net_actor.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
optimizer_c = torch.optim.SGD(ac_net_critic.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer_a = torch.optim.SGD(ac_net_actor.parameters(), lr=0.001, momentum=0.9, nesterov=True)
gamma = 0.99

return_time = 1

N_STEPS = 5000
N_TRAJECTORIES = 2
K_epochs = 5
N_MINI_BATCH = 128
avg_reward = deque(maxlen=100)
avg_STD = deque()
avg_critic_loss = deque()
avg_reward_STD = deque()
avg_value_STD = deque()
#avg_reward_batch = deque(maxlen=N_TRAJECTORIES)
def train(i, episodes, train=False, render = False):
    episode_i = 0
    total_i = 0
    while episode_i < episodes: ## START MAIN LOOP
        cur_state = env.reset()
        done = False
        state_batch = []
        discounted_reward = []
        ret = 0
        mu1_q = []
        cur_state_q = []
        next_state_q = []
        reward_q = []
        action_log_prob_q = []
        value_q = []
        advantage_q = []
        done_q = []
        discounted_reward = []
        action_q = []
        i_in_batch = 0
        avg_reward_batch = []
        while i_in_batch < N_STEPS: ## START EPISODE BATCH LOOP
            cur_state = env.reset()
            done=False
            ret = 0
            i_in_episode = 0
            while not done: ## RUN SINGLE EPISODE
                #action_example = env.action_space.sample() # sample an action randomly
                with torch.no_grad():
                    mu, sd = ac_net_actor(torch.tensor(cur_state).unsqueeze(0).float())
                distribution = Normal(mu[0], sd[0])
                action = distribution.sample()
                clamped_action = torch.clamp(action, -1.0, 1.0).data.numpy() 
                 
                for _ in range(6):
                    with torch.no_grad():
                        val_out = ac_net_critic(torch.tensor(cur_state).unsqueeze(0).float())
                        mu, sd = ac_net_actor(torch.tensor(cur_state).unsqueeze(0).float())
                    
                    distribution = Normal(mu[0], sd[0])    
                    next_state, reward, done, info = env.step(clamped_action)

                    reward_i = reward

                    ret += reward

                    cur_state_q.append(cur_state)
                    next_state_q.append(next_state)
                    reward_q.append(float(reward_i))
                    value_q.append(val_out)
                    action_q.append(action.data.numpy())
                    action_log_prob_q.append(distribution.log_prob(action).data.numpy())
                    done_q.append(1-done)

                    cur_state = next_state
                    i_in_episode += 1
                    i_in_batch += 1
                    total_i += 1
                    if i_in_episode % 8 == 0 and episode_i % 5 == 0 and episode_i >= 0:    
                        env.render()
                    if done == True:
                        break
            ##END SINGLE EPISODE
            
            episode_i += 1
            avg_reward.append(ret)
            avg_reward_batch.append(ret)
        ## END EPISODE BATCH LOOP
        cul_reward = 0
        for reward, done, in zip(reversed(reward_q), reversed(done_q)):
            if done == 1:
                cul_reward = cul_reward*gamma + reward
                discounted_reward.insert(0, cul_reward)
            elif done == 0:
                cul_reward = reward
                discounted_reward.insert(0, cul_reward)
            

        ## START UPDATING CRITIC NETWORK


        for reward_i, value_i in zip(discounted_reward, value_q):
            #critic_loss.append(criterion_val(value_i, torch.tensor(reward_i)))
            advantage_q.append(reward_i - value_i.item())
        advantage_q = (advantage_q-np.mean(advantage_q))/(np.std(advantage_q)/2)

        #nn.utils.clip_grad_norm(ac_net_critic.parameters(),0.5)
        optimizer_c.step()

        # FINISH UPDATING CRITIC NETWORK
        
        batch_length = len(cur_state_q)

        cur_state_t = torch.tensor(cur_state_q)
        advantage_t = torch.tensor(advantage_q).float()
        action_log_prob_t = torch.tensor(action_log_prob_q)
        action_t = torch.tensor(action_q)
        reward_t = torch.tensor(discounted_reward)
        epsilon = 0.2

        avg_STD = deque()
        avg_critic_loss = deque()
        avg_reward_STD = deque()
        avg_value_STD = deque()

        for epoch in range(K_epochs):
            indexes = torch.tensor(np.random.permutation(batch_length)).type(torch.LongTensor)
            num_mini_batches = np.ceil(batch_length % N_MINI_BATCH)
            sample_count = 0
            n_batch = 0
            batch_start = 0
            batch_end = 0
            while batch_end < batch_length:
                batch_end = batch_start + N_MINI_BATCH
                if batch_end > batch_length:
                    batch_end = batch_length

                batch_idx = indexes[batch_start:batch_end]

                batch_state_t = torch.index_select(cur_state_t, 0, batch_idx).float()
                batch_advantage_t = torch.index_select(advantage_t, 0, batch_idx)
                batch_action_log_prob_t = torch.index_select(action_log_prob_t, 0, batch_idx)
                batch_action_t = torch.index_select(action_t, 0, batch_idx)
                batch_reward_t = torch.index_select(reward_t, 0, batch_idx)

                batch_start = batch_end

                n_batch += 1
                
                if batch_state_t.size()[0] == 0:
                    print("stop right there")

                new_val = ac_net_critic(batch_state_t)
                critic_loss_batch = criterion_val(new_val, batch_reward_t.unsqueeze(1))
                optimizer_c.zero_grad()
                critic_loss_batch.backward()
                nn.utils.clip_grad_norm(ac_net_critic.parameters(),1)
                optimizer_c.step()

                mu_batch, sd_batch = ac_net_actor(batch_state_t)
                
                batch_distribution = Normal(mu_batch, sd_batch)
                exp_probs = batch_distribution.log_prob(batch_action_t).exp()
                old_exp_probs = batch_action_log_prob_t.exp()
                r_theta_i = torch.div(exp_probs,old_exp_probs)
                batch_advantage_t4 = batch_advantage_t.unsqueeze(1).expand_as(r_theta_i)

                surrogate1 = r_theta_i*batch_advantage_t4
                surrogate2 = torch.clamp(r_theta_i, 1 - epsilon, 1 + epsilon)*batch_advantage_t4

                r_theta_surrogate_min = torch.min(surrogate1,surrogate2)
                L_clip = -torch.sum(r_theta_surrogate_min)/r_theta_surrogate_min.size()[0]# - 0.00001*torch.mean(batch_distribution.entropy())
                optimizer_a.zero_grad()
                L_clip.backward()
                nn.utils.clip_grad_norm(ac_net_actor.parameters(),1)
                optimizer_a.step()

                avg_STD.append(torch.std(mu_batch[:,0]).item())
                avg_value_STD.append(critic_loss_batch.item())
                avg_reward_STD.append(torch.std(batch_reward_t.unsqueeze(1)[:,0]).item())

                if torch.isnan(ac_net_actor.fc1.weight[0][0]):
                    print("A NAN!!!!!")
                ##print("ok")



        if episode_i % return_time == 0:
            print("%4d, %6.0d, %6.2f, %6.2f, %6.2f, %6.3f" % (episode_i, total_i,np.mean(avg_reward_batch), np.mean(avg_reward),np.mean(avg_value_STD), np.mean(avg_STD)))
        ## END UPDATING ACTOR 
        
    
    return episode_i
    ## END MAIN LOOP

i_global = train(0, 100000, train=True, render=True)
print(i_global)
env.close()