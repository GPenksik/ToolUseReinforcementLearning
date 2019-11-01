# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
OpenAI gym execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import argparse
import json
import logging
import os
import sys
# import time
sys.path.append(os.path.abspath("/home/adf/exp715/EnvTest/Lib/site-packages/"))
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment
# import tensorflow as tf


warnings.filterwarnings("ignore", category=DeprecationWarning)


# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000

# python examples/openai_gym.py CartPole-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 2000 -m 200


def main():

    # start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('run_number', help="Consecutive number of this run")
    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-ns', '--network-size', type=int,default=1024)
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-rl', '--reward-level', type=int, default=3)
    parser.add_argument('-rn', '--random-level', type=int, default=3)
    parser.add_argument('-sc', '--reward-scale', type=int, default=6)
    parser.add_argument('-rp', '--repeat', type=int, default=1, help='How many times to repeat an action')
    parser.add_argument('-bt', '--batch-size', type=int)
    parser.add_argument('-mc', '--memory-capacity', type=int)
    parser.add_argument('-os', '--optimization-steps', type=int)
    parser.add_argument('-bs', '--baseline-steps', type=int)
    parser.add_argument('-sf', '--subsampling-fraction', type=float,default=0.9)
    parser.add_argument('-lr', '--likelihood-ratio', type=float, default=0.1)
    parser.add_argument('-sd', '--seed', type=int, default=None, help='Random seed for this trial')
    parser.add_argument('-tk', '--task', type=int, default=0)

    args = parser.parse_args()

    # SET BASIC PARAMETERS

    random_seed = args.seed
    agent_save_period = args.save_episodes
    visualize_period = 1
    run_number = args.run_number

    if args.load:
        load_agent = True
        agent_filename = args.load
    else:
        load_agent = False

    if False:
        to_visualize = True
    else:
        to_visualize = False

    # Set logging level
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    environment = Environment.create(environment='gym', level='EnvTestContinuousR-v2', visualize=False)

    # Set random seed, reward level and random level for environment
    environment.environment.env.seed(random_seed)
    environment.environment.env.set_reward(args.reward_level)
    environment.environment.env.set_random(args.random_level)
    environment.environment.env.set_reward_scale(args.reward_scale)
    environment.environment.env.set_task(args.task)

    # Initialize Agent-Network-Model objects

    with open('/home/adf/exp715/Box2dEnv/examples/configs/{}'.format(args.agent), 'r') as fp:
        agentSpec = json.load(fp=fp)

    agentSpec['optimization_steps'] = args.optimization_steps
    agentSpec['network']['layers'][0]['size'] = args.network_size
    agentSpec['network']['layers'][1]['size'] = args.network_size
    agentSpec['critic_network']['layers'][0]['size'] = args.network_size
    agentSpec['critic_network']['layers'][1]['size'] = args.network_size
    agentSpec['batch_size'] = args.batch_size
    agentSpec['subsampling_fraction']=args.subsampling_fraction
    agentSpec['critic_optimizer']['num_steps'] = args.baseline_steps
    agentSpec['likelihood_ratio_clipping'] = args.likelihood_ratio

    agent = Agent.create(
        max_episode_timesteps=3000,
        agent=agentSpec,
        environment=environment,
        seed=random_seed
    )

    agent.initialize()

    if load_agent:
        agent.restore_model(directory='/home/adf/exp715/Box2dEnv/Box2dEnv/saves/modelSave', file=agent_filename)

    runner = Runner(
        agent=agent,
        environment=environment
    )

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    # Naming variables
    nNum = str(run_number).zfill(3)
    task = environment.environment.env.task
    if task == 'LIFT':
        nTask = 'L'
    else:
        nTask = 'P'
    nReward = environment.environment.env.reward_level
    nRandom = environment.environment.env.rand_level
    nSeed = str(random_seed).zfill(2)
    nAlg = 'PPO'

    nName = ("{}-{}{}{}-{}-{}".format(nNum, nTask, nReward, nRandom, nSeed, nAlg))

    def episode_finished(r, id_=None):

        save_period = 20
        if r.episodes % save_period == 0:
            with open('/home/adf/exp715/Box2dEnv/Box2dEnv/saves/{}.csv'.format(nName), 'a+') as csv:
                for reward in r.episode_rewards[-save_period:]:
                    csv.write("{:2.2f}\n".format(reward))
                # print("Saving, yo!")

        if r.episodes == 1 or (r.episodes % agent_save_period == 0):
            logger.info(
                "\nSaving agent to {} at episode {}".format(
                    '/home/adf/exp715/Box2dEnv/Box2dEnv/saves/{}'.format(nName), r.episodes))


        return True

    runner.run(
        num_episodes=args.episodes, num_timesteps=args.timesteps, max_episode_timesteps=args.max_episode_timesteps, num_repeat_actions=1,
        # Callback
        callback=episode_finished, callback_episode_frequency=1, callback_timestep_frequency=None,
        # Tqdm
        use_tqdm=True, mean_horizon=100,
        # Evaluation
        evaluation=False, evaluation_callback=None, evaluation_frequency=None,
        max_evaluation_timesteps=None, num_evaluation_iterations=0
    )

    runner.close()

    #logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))
    #logger.info("Time taken = {}".format( time.time()-start_time))

if __name__ == '__main__':
    main()
