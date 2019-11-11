import sys
import os
from collections import deque
from torch.distributions import Normal
import warnings
import argparse
sys.path.append(os.path.abspath("/home/adf/exp715/EnvTest/Lib/site-packages/"))
import gym
import torch
import torch.nn as nn
import numpy as np
import Curious_net_actor_cont as Net_Actor
import Curious_net_critic_cont as Net_Critic
import random
np.set_printoptions(precision=3, linewidth=200, floatmode='fixed', suppress=True)
torch.set_printoptions(precision=3)
device = torch.device("cpu")
torch.set_default_tensor_type('torch.FloatTensor')

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('run_number', help="Consecutive number of this run")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-ns', '--network-size', type=int,default=128)
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-rl', '--reward-level', type=int, default=3)
    parser.add_argument('-rn', '--random-level', type=int, default=3)
    parser.add_argument('-sc', '--reward-scale', type=int, default=6)
    parser.add_argument('-rp', '--repeat', type=int, default=1, help='How many times to repeat an action')
    parser.add_argument('-bt', '--batch-size', type=int)
    parser.add_argument('-os', '--optimization-steps', type=int)
    parser.add_argument('-bs', '--baseline-steps', type=int)
    parser.add_argument('-mb', '--mini-batch', type=int, default=128)
    parser.add_argument('-sd', '--seed', type=int, default=None, help='Random seed for this trial')
    parser.add_argument('-tk', '--task', type=int, default=0)
    parser.add_argument('-gm', '--gamma', type=float, default=0.99)
    parser.add_argument('-lr', '--epsilon', type=float, default=0.2)

    args = parser.parse_args()

    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(a=random_seed)

    # Make environment and set parameters
    env = gym.make('EnvTestContinuousR-v2')
    env.unwrapped.set_reward(args.reward_level)
    env.unwrapped.set_random(args.random_level)
    env.unwrapped.set_reward_scale(args.reward_scale)
    env.unwrapped.set_task(args.task)
    env.unwrapped.seed(random_seed)
    env.unwrapped.set_repeat(args.repeat)

    return_time = 1

    # Set network parameters and initialize
    N_STATES = 5
    N_ACTIONS = 3

    # Initialise network and hyper params
    NETWORK_SIZE = args.network_size
    ac_net_critic = Net_Critic.Net(N_STATES, NETWORK_SIZE)
    ac_net_actor = Net_Actor.Net(N_STATES, N_ACTIONS, NETWORK_SIZE)

    criterion_val = nn.MSELoss()
    # optimizer_c = torch.optim.Adam(ac_net_critic.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
    # optimizer_a = torch.optim.Adam(ac_net_actor.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
    optimizer_c = torch.optim.SGD(ac_net_critic.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    optimizer_a = torch.optim.SGD(ac_net_actor.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    gamma = args.gamma

    N_TRAJECTORIES = args.batch_size
    K_epochs = args.optimization_steps
    B_epochs = args.baseline_steps
    N_MINI_BATCH = args.mini_batch
    epsilon = args.epsilon

    # Initialize tracking queues
    avg_reward = deque(maxlen=100)

    # Setup filename
    run_number = args.run_number
    # Naming variables
    nNum = str(run_number).zfill(4)
    task = env.unwrapped.task
    if task == 'LIFT':
        nTask = 'L'
    else:
        nTask = 'P'
    nReward = env.unwrapped.reward_level
    nRandom = env.unwrapped.rand_level
    nSeed = str(random_seed).zfill(2)
    nAlg = 'mPPO'

    nName = ("{}-{}{}{}-{}-{}".format(nNum, nTask, nReward, nRandom, nSeed, nAlg))

    # Initialize global counters
    episode_i = 0
    total_i = 0
    episodes = args.episodes
    # noinspection PyCallingNonCallable
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

        #if episode_i > 500:
        #    env.unwrapped.set_repeat(int(args.repeat/2))

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

                #if i_in_episode % 1 == 0 and episode_i % 10 == 0 and episode_i >= 0:
                #    env.render()
                if i_in_episode > args.max_episode_timesteps:
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
        for epoch in range(B_epochs):
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
                surrogate2 = torch.clamp(r_theta_i, 1 - epsilon, 1 + epsilon) * batch_advantage_t4

                # Calculate batch entropy
                batch_entropy = batch_distribution.entropy()
                batch_entropy_loss = torch.mean(batch_entropy)

                # Choose minimum of surrogates and calculate L_clip as final loss function
                r_theta_surrogate_min = torch.min(surrogate1, surrogate2)
                L_clip = -torch.sum(r_theta_surrogate_min) / (r_theta_surrogate_min.size()[0])  # + 0.025 * batch_entropy_loss

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

        with open('/home/adf/exp715/Box2dEnv/Box2dEnv/saves/{}.csv'.format(nName), 'a+') as csv:
            for ret_write in zip(np.asarray(avg_reward_batch)):
                csv.write("{:2.2f}\n".format(ret_write[0]))

        # END UPDATE OF BATCH - RETURN TO TOP WHILE STILL EPISODES TO GO

    # END MAIN LOOP

    env.close()

if __name__ == '__main__':
    main()
