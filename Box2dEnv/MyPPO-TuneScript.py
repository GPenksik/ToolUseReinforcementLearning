import sys
import os
import gym
import torch
import torch.nn as nn
import numpy as np
import Curious_net_actor_cont_3 as Net_Actor
import Curious_net_critic_cont as Net_Critic
from collections import deque
from torch.distributions import Normal
import random
sys.path.append(os.path.abspath("C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv"))
sys.path.append(os.path.abspath("C:\\Users\\genia\\AppData\\Local\\conda\\conda\\envs\\EnvTest\\Lib\\site-packages"))


np.set_printoptions(precision=3, linewidth=200, floatmode='fixed', suppress=True)
torch.set_printoptions(precision=3)
device = torch.device("cpu")
torch.set_default_tensor_type('torch.FloatTensor')

# TODO Wrap in ARGS HERE

# TODO set seed args
random_seed = 20
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(a=random_seed)

# Make environment and set parameters
# TODO Move gym params here and set args
env = gym.make('EnvTestContinuousR-v2')
env.unwrapped.set_reward(2)
env.unwrapped.set_random(3)
env.unwrapped.set_task(1)
env.unwrapped.seed(random_seed)
env.unwrapped.set_repeat(1)

load = False
return_time = 1

# Set network parameters and initialize
# TODO set args
N_STATES = 5
N_ACTIONS = 3

# Initialise network and hyper params
# TODO set args
ac_net_critic = Net_Critic.Net(N_STATES, 196)
ac_net_actor = Net_Actor.Net(N_STATES, N_ACTIONS, 196)

criterion_val = nn.MSELoss()
optimizer_c = torch.optim.Adam(ac_net_critic.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
optimizer_a = torch.optim.Adam(ac_net_actor.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
# optimizer_c = torch.optim.SGD(ac_net_critic.parameters(), lr=0.001, momentum=0.9, nesterov=True)
# optimizer_a = torch.optim.SGD(ac_net_actor.parameters(), lr=0.001, momentum=0.9, nesterov=True)

# TODO Set optimizer args
gamma = 0.95

# N_STEPS = 6000
N_TRAJECTORIES = 50
K_EPOCHS = 4
B_EPOCHS = 1
N_MINI_BATCH = 2000
EPSILON = 0.2
N_ACTIONS_PER_EPISODE = 500

# Initialize tracking queus
avg_reward = deque(maxlen=100)
avg_curious_reward = deque(maxlen=100)


# noinspection PyCallingNonCallable
def train(episodes):
    # Initialize global counters
    first_batch = True
    episode_i = 0
    total_i = 0

    while episode_i < episodes:  # START MAIN LOOP
        # Initialize batch lists
        current_state_q = []
        next_state_q = []
        reward_q = []
        action_log_prob_q = []
        value_q = []
        advantage_q_new = []
        done_q = []
        action_q = []
        avg_reward_batch = []
        episode_in_batch = 0
        i_in_batch = 0

        # while i_in_batch < N_STEPS:  # START EPISODE BATCH LOOP
        while episode_in_batch < N_TRAJECTORIES:
            # Reset environment and get first state
            cur_state = env.reset()
            done = False
            ret = 0
            i_in_episode = 0

            while not done:  # RUN SINGLE EPISODE
                # Get parameters for distribution and assign action
                torch_state = torch.tensor(cur_state).unsqueeze(0).float()

                with torch.no_grad():
                    mu, sd = ac_net_actor(torch_state)
                    val_out = ac_net_critic(torch_state)

                distribution = Normal(mu[0], sd[0])
                action = distribution.sample()
                clamped_action_t = torch.clamp(action, -1.0, 1.0)
                clamped_action = clamped_action_t.data.numpy()

                for action_count in range(10):
                    # Step environment
                    next_state, reward, done, info = env.step(clamped_action)

                    # Append values to queues
                    current_state_q.append(cur_state)
                    next_state_q.append(next_state)
                    reward_q.append(float(reward))
                    value_q.append(val_out)
                    action_q.append(clamped_action)
                    action_log_prob_q.append(distribution.log_prob(clamped_action_t).data.numpy())
                    done_q.append(1-done)

                    ret += reward  # Sum total reward for episode

                    # Iterate counters, etc
                    cur_state = next_state
                    i_in_episode += 1
                    i_in_batch += 1
                    total_i += 1

                    if i_in_episode % 10 == 0 and episode_i % 25 == 0 and episode_i >= 0:
                        env.render()

                    # TODO get args
                    if i_in_episode > 3000:
                        done = True
                    if done:
                        break

            # END SINGLE EPISODE

            episode_in_batch += 1
            episode_i += 1
            avg_reward.append(ret)
            avg_reward_batch.append(ret)

        # END EPISODE BATCH LOOP


        # START CUMULATIVE REWARD CALC
        discounted_reward = []
        cumul_reward = 0
        for reward, done, in zip(reversed(reward_q), reversed(done_q)):
            if done == 1:
                cumul_reward = cumul_reward*gamma + reward
                discounted_reward.insert(0, cumul_reward)
            elif done == 0:
                cumul_reward = reward
                discounted_reward.insert(0, cumul_reward)

        # SET UP TENSORS
        batch_length = len(current_state_q)

        current_state_t = torch.tensor(current_state_q).float()
        action_log_prob_t = torch.tensor(action_log_prob_q).float()
        action_t = torch.tensor(action_q).float()
        reward_t = torch.tensor(discounted_reward).float()

        # CALCULATE ADVANTAGE
        value_t_new = ac_net_critic(current_state_t)
        for reward_i, value_i in zip(np.asarray(discounted_reward), value_t_new.data.numpy()):
            advantage_q_new.append(reward_i - value_i)
        advantage_q_new = np.asarray(advantage_q_new)
        # TODO check how this is converted between numpy and tensor

        advantage_q_new = (advantage_q_new-np.mean(advantage_q_new))/(np.std(advantage_q_new))

        advantage_t = torch.tensor(advantage_q_new).float()

        # START UPDATING NETWORKS

        # START BASELINE OPTIMIZE
        for epoch in range(B_EPOCHS):
            # Get random permutation of indexes
            indexes = torch.tensor(np.random.permutation(batch_length)).type(torch.LongTensor)
            n_batch = 0
            batch_start = 0
            batch_end = 0

            # Loop over permutation
            while batch_end < batch_length:
                # Get batch indexes
                batch_end = batch_start + N_MINI_BATCH
                if batch_end > batch_length:
                    batch_end = batch_length

                batch_idx = indexes[batch_start:batch_end]

                # Gather data from saved tensors
                batch_state_t = torch.index_select(current_state_t, 0, batch_idx).float()
                batch_reward_t = torch.index_select(reward_t, 0, batch_idx)

                # Get new baseline values
                new_val = ac_net_critic(batch_state_t)

                # Calculate loss compared with reward and optimize
                critic_loss_batch = criterion_val(new_val, batch_reward_t.unsqueeze(1))

                # Do optimization
                optimizer_c.zero_grad()
                critic_loss_batch.backward()
                optimizer_c.step()

                # Iterate counters
                batch_start = batch_end
                n_batch += 1
        # END BASELINE OPTIMIZE

        # START POLICY OPTIMIZE
        for epoch in range(K_EPOCHS):
            # Get random permutation of indexes
            indexes = torch.tensor(np.random.permutation(batch_length)).type(torch.LongTensor)
            n_batch = 0
            batch_start = 0
            batch_end = 0

            # Loop over permutation
            while batch_end < batch_length:
                # Get batch indexes
                batch_end = batch_start + N_MINI_BATCH
                if batch_end > batch_length:
                    batch_end = batch_length

                batch_idx = indexes[batch_start:batch_end]

                # Gather data from saved tensors
                batch_state_t = torch.index_select(current_state_t, 0, batch_idx).float()
                batch_advantage_t = torch.index_select(advantage_t, 0, batch_idx).float()
                batch_action_log_prob_t = torch.index_select(action_log_prob_t, 0, batch_idx)
                batch_action_t = torch.index_select(action_t, 0, batch_idx)
                # batch_reward_t = torch.index_select(reward_t, 0, batch_idx)

                # Get new batch of parameters and action log probs
                mu_batch, sd_batch = ac_net_actor(batch_state_t)
                batch_distribution = Normal(mu_batch, sd_batch)
                exp_probs = batch_distribution.log_prob(batch_action_t).exp()
                old_exp_probs = batch_action_log_prob_t.exp()
                r_theta_i = torch.div(exp_probs, old_exp_probs)

                # Expand advantage to dimensions of r_theta_i
                batch_advantage_t4 = batch_advantage_t.expand_as(r_theta_i)

                # Calculate the options
                surrogate1 = r_theta_i * batch_advantage_t4
                surrogate2 = torch.clamp(r_theta_i, 1 - EPSILON, 1 + EPSILON) * batch_advantage_t4
                batch_entropy = batch_distribution.entropy()
                batch_entropy_loss = torch.mean(torch.pow(batch_entropy, 2))

                # Choose minimum of surrogates and calculate L_clip as final loss function
                r_theta_surrogate_min = torch.min(surrogate1, surrogate2)
                L_clip = -torch.sum(r_theta_surrogate_min) / r_theta_surrogate_min.size()[0] + 0.03 * batch_entropy_loss

                # if batch_entropy_loss > 1.2:
                #     L_clip = L_clip + 0.05 * batch_entropy_loss

                # Optimize
                optimizer_a.zero_grad()
                L_clip.backward()
                optimizer_a.step()

                # Iterate counters
                batch_start = batch_end
                n_batch += 1
        # END UPDATING ACTOR

        if episode_i % return_time == 0:
            print("%4d, %6.0d, %6.2f, %6.2f | %6.2f"
                  % (episode_i, total_i, np.mean(avg_reward_batch), np.mean(avg_reward), torch.mean(batch_entropy).item()))

        with open('C:\\Users\\genia\\source\\repos\\Box2dEnv\\Box2dEnv\\saves\\{}.csv'.format("testWrite"), 'a+') as csv:
            for ret_write in zip(np.asarray(avg_reward_batch)):
                csv.write("{:2.2f}\n".format(ret_write[0]))

    return episode_i
    # END MAIN LOOP


i_global = train(1000)
env.close()
