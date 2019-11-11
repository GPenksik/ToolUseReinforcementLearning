import sys
import os
import gym
import torch
import torch.nn as nn
import numpy as np
import Curious_net_actor_cont as Net_Actor
import Curious_net_critic_cont as Net_Critic
import Curious_net_rnd as Net_rnd
from collections import deque
from torch.distributions import Normal

sys.path.append(os.path.abspath("C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv"))

env = gym.make('MountainCarContinuous-v0')

load = False

N_STATES = 2
N_ACTIONS = 1
np.set_printoptions(precision=3, linewidth=200, floatmode='fixed', suppress=True)
torch.set_printoptions(precision=3)

# Initialise network and hyper params
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type('torch.FloatTensor')
N_HIDDEN = 64
N_HIDDEN_RND = 16
ac_net_critic = Net_Critic.Net(N_STATES, N_HIDDEN)
ac_net_actor = Net_Actor.Net(N_STATES, N_ACTIONS, N_HIDDEN)
ac_net_rnd = Net_rnd.Net(6, N_HIDDEN_RND)
ac_net_pred = Net_rnd.Net(6, int(N_HIDDEN_RND/2))

criterion_val = nn.MSELoss()
# optimizer_c = torch.optim.Adam(ac_net_critic.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
# optimizer_a = torch.optim.Adam(ac_net_actor.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
# optimizer_rnd = torch.optim.Adam(ac_net_pred.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)

optimizer_c = torch.optim.SGD(ac_net_critic.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer_a = torch.optim.SGD(ac_net_actor.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer_rnd = torch.optim.SGD(ac_net_pred.parameters(), lr=0.001, momentum=0.0, nesterov=False)

gamma = 0.99

return_time = 1

N_STEPS = 100
# N_STEPS = 500
N_TRAJECTORIES = 12
K_epochs = 5
B_epochs = 5
R_epochs = 1
N_MINI_BATCH = 512
avg_reward = deque(maxlen=100)
avg_curious_reward = deque(maxlen=100)
avg_STD = deque()
avg_critic_loss = deque()
avg_reward_STD = deque()
avg_value_STD = deque()
# ac_net_pred.load_state_dict(ac_net_rnd.state_dict())
# ac_net_pred.out.weight.data = torch.add(ac_net_pred.out.weight.data, 0.1)
# ac_net_pred.fc1.weight.data = torch.add(ac_net_pred.fc1.weight.data, 0.2)
# ac_net_pred.fc2.weight.data = torch.add(ac_net_pred.fc2.weight.data, 0.2)


def clip_min_grad_value_(parameters, clip_value):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = torch.tensor(float(clip_value))
    for p in filter(lambda p: p.grad is not None, parameters):
        # p.grad.data.clamp_(min=-clip_value, max=clip_value)
        # p.grad.data = torch.min(torch.max(p.grad.data, clip_value), -clip_value)
        p.grad.data = torch.gt(torch.abs(p.grad.data), clip_value).float()*p.grad.data


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
        curious_reward_q = []

        i_in_batch = 0
        while i_in_batch < N_STEPS:  # START EPISODE BATCH LOOP
            cur_state = env.reset()
            done = False
            ret = 0
            curious_ret = 0
            i_in_episode = 0
            while not done:  # RUN SINGLE EPISODE
                # Get parameters for distribution and assign action
                torch_state = torch.tensor(cur_state).unsqueeze(0).float()
                curious_state = torch.mul(torch_state+torch.tensor(0.5), 5).unsqueeze(0)
                with torch.no_grad():
                    mu, sd = ac_net_actor(torch_state)
                    val_out, curious_out = ac_net_critic(torch_state)
                    rnd_val = torch.tensor(1)  # ac_net_rnd(curious_state)
                    pred_val = torch.tensor(1)  # ac_net_pred(curious_state)
                distribution = Normal(mu[0], sd[0])
                action = distribution.sample()
                clamped_action = torch.clamp(action, -0.2, 0.2).data.numpy()


                # Step environment
                next_state, reward, done, info = env.step(clamped_action)

                reward_i = reward

                curious_difference = rnd_val - pred_val
                curious_reward = torch.abs(curious_difference)
                curious_reward_norm = curious_reward.item()/curious_reward_std
                # Append values to queues
                curious_reward_q.append(curious_reward_norm)
                cur_state_q.append(cur_state)
                next_state_q.append(next_state)
                reward_q.append(float(reward_i))
                value_q.append(val_out)
                action_q.append(action.data.numpy())
                action_log_prob_q.append(distribution.log_prob(action).data.numpy())
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

            avg_curious_ret = curious_ret/i_in_episode

            # if episode_i % 10 == 0:
            #     print("V: %6.2f, %6.2f" % (ret, avg_curious_ret))
            episode_i += 1
            avg_reward.append(ret)
            avg_curious_reward.append(avg_curious_ret)
            avg_reward_batch.append(ret)

        # END EPISODE BATCH LOOP

        # NORMALIZE CURIOUS REWARD
        if first_batch:
            #first_batch = False
            curious_reward_std = np.std(curious_reward_q)

        # START CUMULATIVE REWARD CALC
        discounted_reward = []
        discounted_curious_reward = []
        cul_reward = 0
        cul_curious_reward = 0
        for reward, cur_reward, done, in zip(reversed(reward_q), reversed(curious_reward_q), reversed(done_q)):
            if done == 1:
                cul_curious_reward = cul_curious_reward*gamma + cur_reward
                cul_reward = cul_reward*gamma + reward
                discounted_reward.insert(0, cul_reward)
                discounted_curious_reward.insert(0, cul_curious_reward)
            elif done == 0:
                cul_reward = reward
                cul_curious_reward = cur_reward
                discounted_reward.insert(0, cul_reward)
                discounted_curious_reward.insert(0, cul_curious_reward)

        # for reward_i, value_i in zip(discounted_reward, value_q):
        #     advantage_q.append(reward_i - value_i.item())
        # advantage_q = (advantage_q-np.mean(advantage_q))/(np.std(advantage_q)/2) # Should advantage be recalculated at each optimize step?

        # START UPDATING NETWORKS

        batch_length = len(cur_state_q)

        cur_state_t = torch.tensor(cur_state_q).float()
        # curious_state_t = cur_state_t[:, 0]
        # curious_state_t = torch.(-5, 5, 0.001).unsqueeze(1).float()
        # curious_state_t = torch.add(torch.mul(curious_state_t, 5), 10)
        # curious_state_t = torch.mul(curious_state_t + torch.tensor(0.5), 5)
        curious_state_t2 = torch.normal(torch.tensor(np.zeros(10000,)), 1).unsqueeze(1).float()
        # curious_state_t = torch.tensor(np.zeros(10000,)).unsqueeze(1).float()
        # curious_state_t = torch.mul(torch.mul(torch.rand(10000), 0), 5).unsqueeze(1).float()
        curious_state_t2 = torch.add(curious_state_t2, 3)
        curious_state_t = torch.tensor(curious_state_t2)
        flip = -1
        for n_i in range(5):
            curious_state_t = torch.cat((curious_state_t, flip*torch.add(curious_state_t2, -(n_i+1))), 1)
            flip = -flip

        action_log_prob_t = torch.tensor(action_log_prob_q).float()
        action_t = torch.tensor(action_q).float()
        reward_t = torch.tensor(discounted_reward).float()
        curious_reward_t = torch.tensor(discounted_curious_reward).float()
        summed_reward_t = torch.add(curious_reward_t, reward_t)
        epsilon = 0.2

        # curious_state_t = torch.div(torch.mul(curious_state_t, 30).int().float(), 30)

        # START BASELINE OPTIMIZE
        # avg_baseline_loss = []
        for epoch in range(B_epochs):
            # Get random permutation of indexes
            indexes = torch.tensor(np.random.permutation(batch_length)).type(torch.LongTensor)
            n_batch = 0
            batch_start = 0
            batch_end = 0
            # Loop over permutation
            avg_baseline_batch_loss = []
            while batch_end < batch_length:
                # Get batch indexes
                batch_end = batch_start + N_MINI_BATCH
                if batch_end > batch_length:
                    batch_end = batch_length

                batch_idx = indexes[batch_start:batch_end]

                # Gather data from saved tensors
                batch_state_t = torch.index_select(cur_state_t, 0, batch_idx).float()
                batch_reward_t = torch.index_select(reward_t, 0, batch_idx)
                batch_summed_reward_t = torch.index_select(summed_reward_t, 0, batch_idx)
                batch_start = batch_end

                n_batch += 1

                # Get new baseline values
                new_val, new_curious_val = ac_net_critic(batch_state_t)
                # Calculate loss compared with reward and optimize
                # NEEDS TO BE OPTIMZIED WITH CURIOUS VAL AS WELL
                new_summed_val = new_val + new_curious_val
                # critic_loss_batch = criterion_val(new_val, batch_reward_t.unsqueeze(1))
                critic_loss_batch = criterion_val(new_summed_val, batch_summed_reward_t.unsqueeze(1))
                optimizer_c.zero_grad()
                critic_loss_batch.backward()
                optimizer_c.step()

                # avg_value_STD.append(critic_loss_batch.item())
                avg_baseline_batch_loss.append(critic_loss_batch.item())

            # print(np.mean(avg_baseline_batch_loss), " ", end="")

            # avg_baseline_loss.append(np.mean())
        # END BASELINE OPTIMIZE

        # CALCULATE ADVANTAGE
        # Why is this a loop, dumbass?
        curious_advantage_q_new = []
        value_t_new, curious_value_t_new = ac_net_critic(cur_state_t)
        for reward_i, value_i in zip(np.asarray(discounted_reward), value_t_new.data.numpy()):
            advantage_q_new.append(reward_i - value_i)
        advantage_q_new = np.asarray(advantage_q_new)
        for reward_i, value_i in zip(np.asarray(discounted_curious_reward), curious_value_t_new.data.numpy()):
            curious_advantage_q_new.append(reward_i - value_i)
        curious_advantage_q_new = np.asarray(curious_advantage_q_new)

        # advantage_q_new = (advantage_q_new-np.mean(advantage_q_new))/(np.std(advantage_q_new)/2)  # Should advantage be recalculated at each optimize step?
        # curious_advantage_q_new = (curious_advantage_q_new-np.mean(curious_advantage_q_new))/(np.std(curious_advantage_q_new)/2)  # Should advantage be recalculated at each optimize step?

        advantage_t = torch.tensor(advantage_q_new).float()
        curious_advantage_t = torch.tensor(curious_advantage_q_new).float()
        summed_advantage_t = torch.add(advantage_t, curious_advantage_t)

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
                # batch_advantage_t = torch.index_select(advantage_t, 0, batch_idx)
                batch_advantage_t = torch.index_select(curious_advantage_t, 0, batch_idx)
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
                # optimizer_a.step()

        # START CURIOUS OPTIMIZE
        if first_batch:
            n_mini_batch = batch_length
            # batch_state_t_const = batch_state_t
            cur_state_t_const = cur_state_t

        n_mini_batch = len(curious_state_t)
        # batch_length = n_mini_batch
        avg_curious_loss = []
        for epoch in range(R_epochs):
            # Get random permutation of indexes
            indexes = torch.tensor(np.random.permutation(n_mini_batch)).type(torch.LongTensor)
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
                # batch_reward_t = torch.index_select(reward_t, 0, batch_idx)
                # batch_summed_reward_t = torch.index_select(summed_reward_t, 0, batch_idx)
                batch_start = batch_end
                n_batch += 1

                with torch.no_grad():
                    rnd_val = ac_net_rnd(batch_state_t)
                pred_val = ac_net_pred(batch_state_t)
                # Calculate loss compared with reward and optimize
                pred_loss_batch_curious = criterion_val(pred_val, rnd_val)
                optimizer_rnd.zero_grad()
                pred_loss_batch_curious.backward()
                # nn.utils.clip_grad_norm(ac_net_pred.parameters(), 1)
                nn.utils.clip_grad_value_(ac_net_pred.parameters(), 100)
                # clip_min_grad_value_(ac_net_pred.parameters(), 0.2)
                #nn.utils.clip_grad_norm(ac_net_pred.parameters(), 5)

                optimizer_rnd.step()
                avg_curious_loss.append(pred_loss_batch_curious.item())

            #print((pred_loss_batch_curious.data.numpy()), " ", end="")
            # print("")
            # print(epoch)
        #print("")

        # End curious optimization
        if first_batch:
            test_state2 = torch.arange(0, 5, 0.49).unsqueeze(1).float()
            test_state = torch.tensor(test_state2)
            flip = -1
            for n_i in range(5):
                test_state = torch.cat((test_state, flip * torch.add(test_state2, -(n_i + 1))), 1)
                flip = -flip

        with torch.no_grad():
            test_rnd_val = ac_net_rnd(test_state)
            test_pred_val = ac_net_pred(test_state)

        if first_batch:
            test_rewards_norm = torch.abs(test_rnd_val - test_pred_val)  # - test_pred_val)
            first_batch = False

        test_rewards = torch.abs(test_rnd_val - test_pred_val) / test_rewards_norm
        # test_rewards = test_pred_val
        print(test_rewards.transpose(0, 1).data.numpy())



        # if episode_i % return_time == 0:
        #     print("%4d, %6.0d, %6.2f, %6.2f, %6.2f"
        #           % (episode_i, total_i, np.mean(avg_reward_batch), np.mean(avg_reward), np.mean(curious_reward_q)))
        # END UPDATING ACTOR

    return episode_i
    # END MAIN LOOP




i_global = train(2000)
env.close()
