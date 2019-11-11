import sys
import os
import gym
import torch
import torch.nn as nn
import numpy as np
import Curious_net_actor_cont as Net_Actor
import Curious_net_critic_cont as Net_Critic
import Curious_net_rnd_conv as Net_rnd
from collections import deque
from torch.distributions import Normal

sys.path.append(os.path.abspath("C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv"))

env = gym.make('MountainCarContinuous-v0')

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
N_HIDDEN = 16
N_HIDDEN_RND = 32
N_CHANNELS_RND = 32

ac_net_critic = Net_Critic.Net(N_STATES, N_HIDDEN)
ac_net_actor = Net_Actor.Net(N_STATES, N_ACTIONS, N_HIDDEN)
ac_net_c_critic = Net_Critic.Net(N_STATES, N_HIDDEN)
ac_net_rnd = Net_rnd.Net(N_CURIOUS_STATES, N_CHANNELS_RND, N_HIDDEN_RND)
ac_net_pred = Net_rnd.Net(N_CURIOUS_STATES, N_CHANNELS_RND*2, N_HIDDEN_RND)

criterion_val = nn.SmoothL1Loss()

optimizer_c = torch.optim.SGD(ac_net_critic.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer_cc = torch.optim.SGD(ac_net_c_critic.parameters(), lr=0.001, momentum=0.9, nesterov=True)

optimizer_a = torch.optim.SGD(ac_net_actor.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer_rnd = torch.optim.SGD(ac_net_pred.parameters(), lr=0.0005, momentum=0.0, nesterov=False)

gamma1 = 0.95
gamma2 = 0.99

return_time = 1

N_STEPS = 10000
# N_STEPS = 500
N_TRAJECTORIES = 12
K_epochs = 8
B_epochs = 1
R_epochs = 1
N_MINI_BATCH = 512
N_MINI_BATCH2 = 512
epsilon = 0.3
N_CURIOUS_BATCH = 256

avg_reward = deque(maxlen=20)
avg_curious_reward = deque(maxlen=20)
avg_max_height = deque(maxlen=20)
avg_STD = deque()
avg_critic_loss = deque()
avg_reward_STD = deque()
avg_value_STD = deque()

p1 = np.random.normal(0, 5, (N_CURIOUS_STATES, 1))
p2 = np.random.normal(0, 15, (N_CURIOUS_STATES, 1))


def get_curious_state(curious_state, p1i, p2i):
    curious_state_t_new = np.zeros((len(curious_state), N_CURIOUS_STATES))
    for x, p1x, p2x, p1y, p2y in zip(range(N_CURIOUS_STATES), p1i, p2i, reversed(p1i), reversed(p2i)):
        curious_state_t_new[:, x] = np.squeeze(
            p1x * np.cos(p2x * (-curious_state)) + p1y * np.sin(p2y * (-curious_state)))
    return torch.tensor(curious_state_t_new).float()


# noinspection PyCallingNonCallable
def train(episodes):
    first_batch = True
    episode_i = 0
    total_i = 0
    curious_reward_std = 0.2
    while episode_i < episodes:  # START MAIN LOOP
        cur_state_q = []
        next_state_q = []
        reward_q = []
        action_log_prob_q = []
        value_q = []
        advantage_q_new = []
        done_q = []
        action_q = []
        avg_reward_batch = []
        avg_curious_reward_batch = []
        curious_reward_q = []

        i_in_batch = 0
        while i_in_batch < N_STEPS:  # START EPISODE BATCH LOOP
            cur_state = env.reset()
            done = False
            ret = 0
            curious_ret = 0
            i_in_episode = 0
            max_episode_distance = []
            while not done:  # RUN SINGLE EPISODE
                # Get parameters for distribution and assign action
                torch_state = torch.tensor(cur_state).unsqueeze(0).float()
                with torch.no_grad():
                    mu, sd = ac_net_actor(torch_state)
                    val_out = ac_net_critic(torch_state)
                    curious_out = ac_net_c_critic(torch_state)
                distribution = Normal(mu[0], sd[0])
                action = distribution.sample()
                if episode_i < 15:
                    clamped_action = torch.clamp(action, -1, 1).data.numpy()
                else:
                    clamped_action = torch.clamp(action, -1, 1).data.numpy()

                max_episode_distance.append(cur_state[0])
                # Step environment
                next_state, reward, done, info = env.step(clamped_action)
                next_torch_state = torch.tensor(next_state).unsqueeze(0).float()
                curious_state = get_curious_state(next_torch_state.data.numpy()[:, 0], p1, p2)

                with torch.no_grad():
                    rnd_val = ac_net_rnd(curious_state.unsqueeze(1))
                    pred_val = ac_net_pred(curious_state.unsqueeze(1))

                curious_reward = torch.pow((rnd_val - pred_val), 2)
                curious_reward = np.mean(curious_reward.data.numpy())
                curious_reward_norm = curious_reward
                # Append values to queues
                curious_reward_q.append(curious_reward)
                cur_state_q.append(cur_state)
                next_state_q.append(next_state)
                reward_i = reward
                reward_q.append(float(reward_i))
                # value_q.append(val_out)
                action_q.append(action.data.numpy())
                action_log_prob_q.append(distribution.log_prob(torch.tensor(clamped_action)).data.numpy())
                done_q.append(1-done)  # Why 1-done?

                ret += reward  # Sum total reward for episode
                curious_ret += curious_reward_norm

                # Iterate counters, etc
                cur_state = next_state
                i_in_episode += 1
                i_in_batch += 1
                total_i += 1
                if i_in_episode % 50 == 0 and episode_i % 10 == 0 and episode_i >= 0:
                    env.render()
                # if i_in_episode > 500:
                #     done = True
                if done:
                    break

            # END SINGLE EPISODE
            print("%4d, %6.2f, %6.0f | " % (episode_i, np.max(max_episode_distance), curious_ret))
            avg_curious_ret = curious_ret/i_in_episode

            # if episode_i % 10 == 0:
            #     print("V: %6.2f, %6.2f" % (ret, avg_curious_ret))
            episode_i += 1
            avg_reward.append(ret)
            avg_curious_reward.append(curious_ret)
            avg_reward_batch.append(ret)
            avg_curious_reward_batch.append(curious_ret)
            avg_max_height.append(np.max(max_episode_distance))
        # print("")
        # END EPISODE BATCH LOOP

        cur_state_t = torch.tensor(cur_state_q).float()
        max_achieved_height = torch.max(cur_state_t[:, 0]).data.numpy()
        curious_state_t = cur_state_t[:, 0].unsqueeze(1).float()
        curious_state_t = curious_state_t.data.numpy()
        curious_state_t = get_curious_state(curious_state_t, p1, p2)

        # NORMALIZE CURIOUS REWARD
        if first_batch:
            with torch.no_grad():
                test_rnd_val = ac_net_rnd(curious_state_t.unsqueeze(1))
                test_pred_val = ac_net_pred(curious_state_t.unsqueeze(1))

            test_rewards = torch.pow((test_rnd_val - test_pred_val), 2)  # / test_rewards_norm
            test_rewards = test_rewards.data.numpy()
            curious_reward_std = np.std(test_rewards)
            # test_rewards = np.mean(test_rewards)

            first_batch = False

        with torch.no_grad():
            test_rnd_val = ac_net_rnd(curious_state_t.unsqueeze(1))
            test_pred_val = ac_net_pred(curious_state_t.unsqueeze(1))

        test_rewards = torch.pow((test_rnd_val - test_pred_val), 2)
        test_rewards = np.mean(test_rewards.data.numpy())/curious_reward_std

        # START CUMULATIVE REWARD CALC
        curious_reward_q = curious_reward_q / (curious_reward_std*50)
        discounted_reward = []
        discounted_curious_reward = []
        cul_reward = 0
        cul_curious_reward = 0
        for reward, cur_reward, done, in zip(reversed(reward_q), reversed(curious_reward_q), reversed(done_q)):
            if done == 1:
                cul_curious_reward = cul_curious_reward*gamma2 + cur_reward
                cul_reward = cul_reward*gamma1 + reward
                discounted_reward.insert(0, cul_reward)
                discounted_curious_reward.insert(0, cul_curious_reward)
            elif done == 0:
                cul_reward = reward
                cul_curious_reward = cur_reward*gamma2 + reward
                discounted_reward.insert(0, cul_reward)
                discounted_curious_reward.insert(0, cul_curious_reward)

        # CALCULATE ADVANTAGE
        # Why is this a loop, dumbass?
        curious_advantage_q_new = []
        value_t_new = ac_net_critic(cur_state_t)
        curious_value_t_new = ac_net_c_critic(cur_state_t)
        for reward_i, value_i in zip(np.asarray(discounted_reward), value_t_new.data.numpy()):
            advantage_q_new.append(reward_i - value_i)
        advantage_q_new = np.asarray(advantage_q_new)
        for reward_i, value_i in zip(np.asarray(discounted_curious_reward), curious_value_t_new.data.numpy()):
            curious_advantage_q_new.append(reward_i - value_i)
        curious_advantage_q_new = np.asarray(curious_advantage_q_new)



        # advantage_q_new = (advantage_q_new-np.mean(advantage_q_new))/(np.std(advantage_q_new))  # Should advantage be recalculated at each optimize step?
        curious_advantage_q_new = (curious_advantage_q_new-np.mean(curious_advantage_q_new))/(np.std(curious_advantage_q_new))  # Should advantage be recalculated at each optimize step?
        advantage_q_new = (curious_advantage_q_new-np.mean(curious_advantage_q_new))/(np.std(curious_advantage_q_new))  # Should advantage be recalculated at each optimize step?


        #advantage_q_new = (advantage_q_new) / (np.std(advantage_q_new))  # Should advantage be recalculated at each optimize step?
        # curious_advantage_q_new = (curious_advantage_q_new) / (np.std(curious_advantage_q_new))  # Should advantage be recalculated at each optimize step?

        max_curious_advantage = np.max(curious_advantage_q_new)
        std_curious_advantage = np.std(curious_advantage_q_new)
        mean_curious_advantage = np.mean(curious_advantage_q_new)

        max_advantage = np.max(advantage_q_new)
        std_advantage = np.std(advantage_q_new)
        mean_advantage = np.mean(advantage_q_new)


        advantage_t = torch.tensor(advantage_q_new).float()
        curious_advantage_t = torch.tensor(curious_advantage_q_new).float()
        a_prop = 0.5
        summed_advantage_t = torch.add(torch.mul(advantage_t, a_prop), torch.mul(curious_advantage_t, (1-a_prop)))


        # START UPDATING NETWORKS

        batch_length = len(cur_state_q)

        action_log_prob_t = torch.tensor(action_log_prob_q).float()
        action_t = torch.tensor(action_q).float()
        reward_t = torch.tensor(discounted_reward).float()
        curious_reward_t = torch.tensor(discounted_curious_reward).float()
        summed_reward_t = torch.add(curious_reward_t, reward_t)

        # START BASELINE OPTIMIZE
        avg_baseline_loss = []
        for epoch in range(B_epochs):
            # Get random permutation of indexes
            indexes = torch.tensor(np.random.permutation(batch_length)).type(torch.LongTensor)
            n_batch = 0
            batch_start = 0
            batch_end = 0
            # Loop over permutation
            avg_baseline_batch_loss = []
            avg_baseline_curious_batch_loss = []
            while batch_end < batch_length:
                # Get batch indexes
                batch_end = batch_start + N_MINI_BATCH
                if batch_end > batch_length:
                    batch_end = batch_length

                batch_idx = indexes[batch_start:batch_end]

                # Gather data from saved tensors
                batch_state_t = torch.index_select(cur_state_t, 0, batch_idx).float()
                batch_reward_t = torch.index_select(reward_t, 0, batch_idx)
                batch_curious_reward_t = torch.index_select(curious_reward_t, 0, batch_idx)
                batch_summed_reward_t = torch.index_select(summed_reward_t, 0, batch_idx)
                batch_start = batch_end

                n_batch += 1

                # Get new baseline values
                new_val = ac_net_critic(batch_state_t)
                new_curious_val = ac_net_c_critic(batch_state_t)
                # Calculate loss compared with reward and optimize
                # NEEDS TO BE OPTIMIZED WITH CURIOUS VAL AS WELL
                # new_summed_val = new_val + new_curious_val
                critic_loss_batch = criterion_val(new_val, batch_reward_t.unsqueeze(1))
                critic_curious_loss_batch = criterion_val(new_curious_val, batch_curious_reward_t.unsqueeze(1))
                # critic_loss_batch = criterion_val(new_summed_val, batch_summed_reward_t.unsqueeze(1))
                # critic_loss_both = critic_curious_loss_batch  # + critic_loss_batch
                optimizer_c.zero_grad()
                optimizer_cc.zero_grad()

                critic_loss_batch.backward()
                critic_curious_loss_batch.backward()
                optimizer_cc.step()
                optimizer_c.step()

                # avg_value_STD.append(critic_loss_batch.item())
                avg_baseline_batch_loss.append(critic_loss_batch.item())
                avg_baseline_curious_batch_loss.append(critic_curious_loss_batch.item())
            # print(np.mean(avg_baseline_batch_loss), np.mean(avg_baseline_curious_batch_loss), " ", end="")
            # avg_baseline_loss.append(np.mean(avg_baseline_batch_loss))

        # print("")
        # END BASELINE OPTIMIZE

        # START POLICY OPTIMIZE
        for epoch in range(K_epochs):
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
                batch_state_t = torch.index_select(cur_state_t, 0, batch_idx).float()
                batch_advantage_t = torch.index_select(advantage_t, 0, batch_idx)
                # batch_advantage_t = torch.index_select(curious_advantage_t, 0, batch_idx)
                # batch_advantage_t = torch.index_select(summed_advantage_t, 0, batch_idx)

                batch_action_log_prob_t = torch.index_select(action_log_prob_t, 0, batch_idx)
                batch_action_t = torch.index_select(action_t, 0, batch_idx)
                # batch_reward_t = torch.index_select(reward_t, 0, batch_idx)

                batch_start = batch_end
                n_batch += 1

                # Get new batch of parameters and action log probs
                mu_batch, sd_batch = ac_net_actor(batch_state_t)
                batch_distribution = Normal(mu_batch, sd_batch)
                exp_probs = batch_distribution.log_prob(batch_action_t).exp()
                old_exp_probs = batch_action_log_prob_t.exp()
                r_theta_i = torch.div(exp_probs, old_exp_probs)

                # Advantage needs to include curious advantage. Should advantage be recalculated each epoch?
                batch_advantage_t4 = batch_advantage_t.expand_as(r_theta_i)

                surrogate1 = r_theta_i * batch_advantage_t4
                surrogate2 = torch.clamp(r_theta_i, 1 - epsilon, 1 + epsilon) * batch_advantage_t4

                r_theta_surrogate_min = torch.min(surrogate1, surrogate2)
                L_clip = -torch.sum(r_theta_surrogate_min) / r_theta_surrogate_min.size()[0]
                optimizer_a.zero_grad()
                L_clip.backward()
                optimizer_a.step()


        n_mini_batch = len(curious_state_t)
        # batch_length = len(curious_state_t)
        avg_curious_loss = []
        curious_batch_length = N_CURIOUS_BATCH
        for epoch in range(R_epochs):
            # Get random permutation of indexes
            indexes = torch.tensor(np.random.permutation(batch_length)).type(torch.LongTensor)
            # indexes = torch.arange(batch_length).type(torch.LongTensor)
            n_batch = 0
            batch_start = 0
            batch_end = 0
            # Loop over permutation
            # avg_curious_loss = []
            while batch_end < curious_batch_length:
                # Get batch indexes
                batch_end = batch_start + N_CURIOUS_BATCH
                if batch_end > curious_batch_length:
                    batch_end = curious_batch_length

                batch_idx = indexes[batch_start:batch_end]

                # Gather data from saved tensors
                batch_state_t = torch.index_select(curious_state_t, 0, batch_idx).float()
                # batch_reward_t = torch.index_select(reward_t, 0, batch_idx)
                # batch_summed_reward_t = torch.index_select(summed_reward_t, 0, batch_idx)
                batch_start = batch_end
                n_batch += 1

                with torch.no_grad():
                    rnd_val = ac_net_rnd(batch_state_t.unsqueeze(1))
                pred_val = ac_net_pred(batch_state_t.unsqueeze(1))
                # Calculate loss compared with reward and optimize
                pred_loss_batch_curious = criterion_val(pred_val, rnd_val)
                optimizer_rnd.zero_grad()
                pred_loss_batch_curious.backward()
                # nn.utils.clip_grad_norm(ac_net_pred.parameters(), 1)
                # nn.utils.clip_grad_value_(ac_net_pred.parameters(), 100)
                # clip_min_grad_value_(ac_net_pred.parameters(), 0.2)
                #nn.utils.clip_grad_norm(ac_net_pred.parameters(), 5)

                optimizer_rnd.step()
                avg_curious_loss.append(pred_loss_batch_curious.item())

            #print((pred_loss_batch_curious.data.numpy()), " ", end="")
            # print("")
            # print(epoch)
        #print("")


        if episode_i % return_time == 0:
            print("%4d | %6.0d | %6.1f, %6.1f | %6.1f, %6.1f | %6.2f, %6.2f, %6.2f | %6.2f, %6.2f, %6.2f | %6.2f, %6.2f"
                  % (episode_i, total_i,
                     np.mean(avg_reward_batch), np.mean(avg_reward),
                     np.mean(avg_curious_reward_batch), np.mean(avg_curious_reward),
                     max_advantage, mean_advantage, std_advantage,
                     max_curious_advantage, mean_curious_advantage, std_curious_advantage,
                     max_achieved_height, np.mean(avg_max_height)))
        # END UPDATING ACTOR

    return episode_i
    # END MAIN LOOP




i_global = train(2000)
env.close()
