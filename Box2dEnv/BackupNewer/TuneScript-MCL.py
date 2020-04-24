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
import argparse
sys.path.append(os.path.abspath("/home/adf/exp715/EnvTest/Lib/site-packages/"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('run_number', default=999, help="Consecutive number of this run")
    parser.add_argument('gym_id', default='MountainCarContinuous-v0', help="Id of the Gym environment")
    parser.add_argument('-e', '--episodes', type=int, default=1000, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-ns', '--network-size', type=int, default=12)
    parser.add_argument('-cs', '--curious-size', type=int, default=32)
    parser.add_argument('-rs', '--random-seeds', type=int, default=100)
    parser.add_argument('-bt', '--batch-size', type=int, default=1028)
    parser.add_argument('-mc', '--memory-capacity', type=int, default=10000)
    parser.add_argument('-os', '--optimization-steps', type=int, default=10)
    parser.add_argument('-bs', '--baseline-steps', type=int, default=1)
    parser.add_argument('-sf', '--subsampling-fraction', type=int, default=256)
    parser.add_argument('-lr', '--likelihood-ratio', type=float, default=0.1)
    parser.add_argument('-sd', '--seed', type=int, default=None, help='Random seed for this trial')
    parser.add_argument('-gr', '--gamma-reward', type=float, default=0.99)
    parser.add_argument('-gc', '--gamma-curious', type=float, default=0.99)


    args = parser.parse_args()

    # SET BASIC PARAMETERS

    sys.path.append(os.path.abspath("C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv"))

    env = gym.make('MountainCarContinuous-v0')

    load = False

    N_STATES = 2
    N_CURIOUS_STATES = args.random_seeds
    N_ACTIONS = 1
    np.set_printoptions(precision=3, linewidth=200, floatmode='fixed', suppress=True)
    torch.set_printoptions(precision=3)

    # Initialise network and hyper params
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_tensor_type('torch.FloatTensor')
    N_HIDDEN = args.network_size
    N_HIDDEN_RND = args.curious_size
    N_CHANNELS_RND = args.curious_size

    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    ac_net_critic = Net_Critic.Net(N_STATES, N_HIDDEN)
    ac_net_actor = Net_Actor.Net(N_STATES, N_ACTIONS, N_HIDDEN)
    ac_net_c_critic = Net_Critic.Net(N_STATES, N_HIDDEN)
    ac_net_rnd = Net_rnd.Net(N_CURIOUS_STATES, N_CHANNELS_RND, N_HIDDEN_RND)
    ac_net_pred = Net_rnd.Net(N_CURIOUS_STATES, N_CHANNELS_RND, N_HIDDEN_RND)

    criterion_val = nn.SmoothL1Loss()

    optimizer_c = torch.optim.SGD(ac_net_critic.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    optimizer_cc = torch.optim.SGD(ac_net_c_critic.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    optimizer_a = torch.optim.SGD(ac_net_actor.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    optimizer_rnd = torch.optim.SGD(ac_net_pred.parameters(), lr=0.0005, momentum=0.0, nesterov=False)

    gamma1 = args.gamma_reward
    gamma2 = args.gamma_curious

    return_time = 1

    N_STEPS = args.memory_capacity
    # N_STEPS = 500
    N_TRAJECTORIES = 12
    K_epochs = args.optimization_steps
    B_epochs = args.baseline_steps
    R_epochs = 1
    N_MINI_BATCH = args.batch_size
    epsilon = args.likelihood_ratio
    N_CURIOUS_BATCH = args.subsampling_fraction

    avg_reward = deque(maxlen=50)
    avg_curious_reward = deque(maxlen=50)
    avg_max_height_q = deque(maxlen=50)
    avg_STD = deque()
    avg_critic_loss = deque()
    avg_reward_STD = deque()
    avg_value_STD = deque()

    p1 = np.random.normal(0, 5, (N_CURIOUS_STATES, 1))
    p2 = np.random.normal(0, 15, (N_CURIOUS_STATES, 1))

    def get_curious_state(curious_state, p1i, p2i):
        curious_state_t_new = np.zeros((len(curious_state), N_CURIOUS_STATES))
        curious_state_1 = curious_state[:, 0]
        curious_state_2 = curious_state[:, 1]/0.07
        for x, p1x, p2x, p1y, p2y in zip(range(N_CURIOUS_STATES), p1i, p2i, reversed(p1i), reversed(p2i)):
            curious_state_t_new[:, x] = np.squeeze(
                p1x * np.cos(p2x * (-curious_state_1)) + p1y * np.sin(p2y * (-curious_state_1)))
            curious_state_t_new[:, x] += np.squeeze(
                p1x * np.cos(p2x * (-curious_state_2)) + p1y * np.sin(p2y * (-curious_state_2)))
        return torch.tensor(curious_state_t_new).float()


    # START TRAINING
    episodes = args.episodes
    env.env.unwrapped.seed(random_seed)
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
        avg_max_height = []
        i_in_batch = 0
        completed_q = []

        BATCH_REWARD = []
        BATCH_CURIOUS_REWARD = []
        BATCH_MAX_HEIGHT = []

        while i_in_batch < N_STEPS:  # START EPISODE BATCH LOOP
            cur_state = env.reset()
            done = False
            ret = 0
            curious_ret = 0
            i_in_episode = 0
            episode_distance_q = []
            next_cur_state_episode_q = []



            while not done:  # RUN SINGLE EPISODE
                # Get parameters for distribution and assign action
                torch_state = torch.tensor(cur_state).unsqueeze(0).float()
                with torch.no_grad():
                    mu, sd = ac_net_actor(torch_state)
                    # val_out = ac_net_critic(torch_state)
                    # curious_out = ac_net_c_critic(torch_state)
                distribution = Normal(mu[0], sd[0])
                action = distribution.sample()
                if episode_i < 15:
                    clamped_action = torch.clamp(action, -1, 1).data.numpy()
                else:
                    clamped_action = torch.clamp(action, -1, 1).data.numpy()

                episode_distance_q.append(cur_state[0])
                # Step environment
                next_state, reward, done, info = env.step(clamped_action)

                # Append values to queues
                cur_state_q.append(cur_state)
                next_cur_state_episode_q.append(next_state)

                next_state_q.append(next_state)
                reward_i = reward
                reward_q.append(float(reward_i))
                # value_q.append(val_out)
                action_q.append(action.data.numpy())
                action_log_prob_q.append(distribution.log_prob(torch.tensor(clamped_action)).data.numpy())
                done_q.append(1-done)  # Why 1-done?

                ret += reward  # Sum total reward for episode

                # Iterate counters, etc
                cur_state = next_state
                i_in_episode += 1
                i_in_batch += 1
                total_i += 1
                if i_in_episode % 1 == 0 and episode_i % 500 == 0 and episode_i > 0:
                    env.render()
                # if i_in_episode > 500:
                #     done = True
                if done:
                    break

            # END SINGLE EPISODE

            if ret > 0.01:
                completed_q += np.ones((len(episode_distance_q), 1)).tolist()
            else:
                completed_q += np.zeros((len(episode_distance_q), 1)).tolist()

            next_state_episode = np.asarray(next_cur_state_episode_q)
            # next_curious_state = get_curious_state(next_state_episode, p1, p2)
            #
            # with torch.no_grad():
            #     rnd_val = ac_net_rnd(next_curious_state.unsqueeze(1))
            #     pred_val = ac_net_pred(next_curious_state.unsqueeze(1))
            #
            # curious_reward_episode = torch.pow((rnd_val - pred_val), 2)
            # curious_rewards_episode = (curious_reward_episode.data.numpy())

            curious_rewards_episode = completed_q
            curious_reward_q += curious_rewards_episode# .tolist()
            curious_ret = np.sum(curious_rewards_episode)
            avg_curious_ret = curious_ret/i_in_episode

            episode_i += 1
            avg_reward.append(ret)
            avg_curious_reward.append(curious_ret)
            avg_reward_batch.append(ret)
            avg_curious_reward_batch.append(curious_ret)
            avg_max_height_q.append(np.max(episode_distance_q))
            avg_max_height.append(np.max(episode_distance_q))
            print("%4d, %6.2f, %6.0f | " % (episode_i, np.max(episode_distance_q), curious_ret))

            BATCH_CURIOUS_REWARD.append(curious_ret)
            BATCH_MAX_HEIGHT.append(np.max(episode_distance_q))
            BATCH_REWARD.append(ret)

        # print("")
        # END EPISODE BATCH LOOP

        max_achieved_height_in_batch = np.max(avg_max_height)

        # NORMALIZE CURIOUS REWARD
        if first_batch:
            curious_reward_std = np.std(np.asarray(curious_reward_q))
            first_batch = False


        # START CUMULATIVE REWARD CALC
        curious_reward_q = curious_reward_q / curious_reward_std
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
        current_state_t = torch.tensor(cur_state_q).float()
        curious_advantage_q_new = []
        advantage_q_new = []
        with torch.no_grad():
            value_t_new = ac_net_critic(current_state_t)
            curious_value_t_new = ac_net_c_critic(current_state_t)

        for reward_i, value_i in zip(np.asarray(discounted_reward), value_t_new.data.numpy()):
            advantage_q_new.append(reward_i - value_i)
        advantage_q_new = np.asarray(advantage_q_new)
        for reward_i, value_i in zip(np.asarray(discounted_curious_reward), curious_value_t_new.data.numpy()):
            curious_advantage_q_new.append(reward_i - value_i)
        curious_advantage_q_new = np.asarray(curious_advantage_q_new)

        advantage_q_new = (advantage_q_new-np.mean(advantage_q_new))/(np.std(advantage_q_new))  # Should advantage be recalculated at each optimize step?
        curious_advantage_q_new = (curious_advantage_q_new-np.mean(curious_advantage_q_new))/(np.std(curious_advantage_q_new))  # Should advantage be recalculated at each optimize step?
        # curious_advantage_q_new = (np.asarray(discounted_curious_reward) -np.mean(discounted_curious_reward))/(np.std(discounted_curious_reward))  # Should advantage be recalculated at each optimize step?

        max_curious_advantage = np.max(curious_advantage_q_new)
        std_curious_advantage = np.std(curious_advantage_q_new)
        mean_curious_advantage = np.mean(curious_advantage_q_new)

        max_advantage = np.max(advantage_q_new)
        std_advantage = np.std(advantage_q_new)
        mean_advantage = np.mean(advantage_q_new)

        advantage_t = torch.tensor(advantage_q_new).float()
        curious_advantage_t = torch.tensor(curious_advantage_q_new).float()
        completed_t = torch.tensor(np.asarray(completed_q)).float()
        # advantage_t = completed_t * advantage_t
        a_prop = 0.5
        summed_advantage_t = torch.add(torch.mul(advantage_t, 1), torch.mul(curious_advantage_t, 1))

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

                batch_state_t = torch.index_select(current_state_t, 0, batch_idx).float()
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
                batch_state_t = torch.index_select(current_state_t, 0, batch_idx).float()
                if np.max(reward_q) > 0.01:
                    batch_advantage_t = torch.index_select(advantage_t, 0, batch_idx)
                else:
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

        # END OPTIMIZE POLICY

        # START OPTIMIZE CURIOUS

        # curious_state_t = get_curious_state(np.asarray(cur_state_q), p1, p2)
        # avg_curious_loss = []
        # curious_batch_length = N_CURIOUS_BATCH
        # for epoch in range(R_epochs):
        #     # Get random permutation of indexes
        #     indexes = torch.tensor(np.random.permutation(batch_length)).type(torch.LongTensor)
        #     n_batch = 0
        #     batch_start = 0
        #     batch_end = 0
        #     # Loop over permutation
        #     # avg_curious_loss = []
        #     while batch_end < curious_batch_length:
        #         # Get batch indexes
        #         batch_end = batch_start + N_CURIOUS_BATCH
        #         if batch_end > curious_batch_length:
        #             batch_end = curious_batch_length
        #
        #         batch_idx = indexes[batch_start:batch_end]
        #
        #         # Gather data from saved tensors
        #         batch_state_t = torch.index_select(curious_state_t, 0, batch_idx).float()
        #         batch_state_t = batch_state_t.unsqueeze(1)
        #         # batch_reward_t = torch.index_select(reward_t, 0, batch_idx)
        #         # batch_summed_reward_t = torch.index_select(summed_reward_t, 0, batch_idx)
        #         batch_start = batch_end
        #         n_batch += 1
        #
        #         with torch.no_grad():
        #             rnd_val = ac_net_rnd(batch_state_t)
        #         pred_val = ac_net_pred(batch_state_t)
        #         # Calculate loss compared with reward and optimize
        #         optimizer_rnd.zero_grad()
        #         pred_loss_batch_curious = criterion_val(pred_val, rnd_val)
        #         pred_loss_batch_curious.backward()
        #         # nn.utils.clip_grad_norm(ac_net_pred.parameters(), 1)
        #         # nn.utils.clip_grad_value_(ac_net_pred.parameters(), 100)
        #         # clip_min_grad_value_(ac_net_pred.parameters(), 0.2)
        #
        #         optimizer_rnd.step()
        #         avg_curious_loss.append(pred_loss_batch_curious.item())

            # print((pred_loss_batch_curious.data.numpy()), " ", end="")
            # print("")
            # print(epoch)
        # print("")

        # Naming variables
        run_number = args.run_number
        nNum = str(run_number).zfill(3)
        nSeed = str(random_seed).zfill(2)

        nName = ("{}-{}-RT".format(nNum, nSeed))

        if episode_i % return_time == 0:
            print("%4d | %6.0d | %6.1f, %6.1f | %6.1f, %6.1f | %6.2f, %6.2f, %6.2f | %6.2f, %6.2f, %6.2f | %6.2f, %6.2f"
                  % (episode_i, total_i,
                     np.mean(avg_reward_batch), np.mean(avg_reward),
                     np.mean(avg_curious_reward_batch), np.mean(avg_curious_reward),
                     max_advantage, mean_advantage, std_advantage,
                     max_curious_advantage, mean_curious_advantage, std_curious_advantage,
                     max_achieved_height_in_batch, np.mean(avg_max_height_q)))
            with open('/home/adf/exp715/Box2dEnv/Box2dEnv/saves/{}.csv'.format(nName), 'a+') as csv:
                for retw, curious_retw, max_heightw in zip(BATCH_REWARD, BATCH_CURIOUS_REWARD, BATCH_MAX_HEIGHT):
                    csv.write("{:2.2f},{:2.2f},{:2.2f}\n".format(retw, curious_retw, max_heightw))
        # END UPDATING ACTOR

if __name__ == '__main__':
    main()
