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

import json
import logging
import os
import time
import sys

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment
import tensorflow as tf

tf.logging.set_verbosity(v=tf.logging.ERROR)

sys.path.append(os.path.abspath('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\'))
sys.path.append(os.path.abspath('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\examples'))

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    # SET BASIC PARAMETERS
    start_time = time.time()
    random_seed = 21
    agent_save_period = 500
    visualize_period = 1
    run_number = 965

    load_agent = False
    agent_filename = '371-P33-27-PPO-2000'
    to_visualize = False

    # Set logging level
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # if args.import_modules is not None:
    #    for module in args.import_modules.split(','):
    #        importlib.import_module(name=module)

    environment = Environment.create(environment='gym', level='EnvTestContinuousR-v2', visualize=to_visualize)

    # Set random seed for environment
    environment.environment.env.seed(random_seed)
    environment.environment.env.set_reward(3)
    environment.environment.env.set_random(3)
    environment.environment.env.set_reward_scale(6)

    # Initialize Agent-Network-Model objects

    with open('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\examples\\configs\\ppo-new3.json', 'r') as fp:
        agentSpec = json.load(fp=fp)

    with open('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\examples\\configs\\mlp2_network-new.json', 'r') as fp:
        network = json.load(fp=fp)

    # agentSpec['update_mode'].update(batch_size=24)
    # agentSpec['update_mode'].update(frequency=24)
    #agentSpec['baseline']['sizes'] = [512,512]
    agentSpec['optimization_steps'] = 9
    agentSpec['network']['layers'][0]['size'] = 128
    agentSpec['network']['layers'][1]['size'] = 129
    agentSpec['critic_network']['layers'][0]['size'] = 126
    agentSpec['critic_network']['layers'][1]['size'] = 127
    agentSpec['batch_size'] = 13
    agentSpec['subsampling_fraction']=0.8
    agentSpec['critic_optimizer']['num_steps'] = 11
    agentSpec['likelihood_ratio_clipping'] = 0.2

    # network[0].update(size=512)
    # network[1].update(size=512)
    # agentSpec['network']['layers'] = network
    # agentSpec['critic_network']['layers'] = network
    agent = Agent.create(
        max_episode_timesteps=3000,
        agent=agentSpec,
        environment=environment,
        seed=random_seed
        # kwargs=dict(
        #     states=environment.states,
        #     actions=environment.actions,
        #     network=network,
        #     #random_seed=random_seed
    )

    agent.initialize()
    # print("Agent memory ", agent.memory['capacity'])
    # print("Agent baseline steps", agent.baseline_optimizer['num_steps'])
    # print("Agent optimizer steps", agent.optimizer['num_steps'])

    if load_agent:
        agent.restore_model(directory='C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\modelSave',
                            file=agent_filename)

    runner = Runner(
        agent=agent,
        environment=environment
    )

    # logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

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

        # if r.episode == 1:
        # r.agent.restore_model('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\modelSave')

        save_period = 5
        if r.episodes % visualize_period == 0:
            if to_visualize:
                environment.visualize = True  # Set to true to visualize
        else:
            environment.visualize = False

        if r.episodes % save_period == 0:
            with open('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\{}.csv'.format(nName), 'a+') as csv:
                for reward in r.episode_rewards[-save_period:]:
                    csv.write("{:2.2f}\n".format(reward))
                # print("\nSaving, yo!")

        if r.episodes == 1 or (r.episodes % agent_save_period == 0):
            logger.info(
                "\nSaving agent to {} at episode {}".format(
                    'C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\modelSave\\{}'.format(nName), r.episodes))
            # r.agent.save(
            #     directory='C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\modelSave\\{}{}'.format(nName, r.episodes),
            #     append_timestep=False)

        return True

    def episode_finish(r, id_=None):
        print(r)

    runner.run(
        num_episodes=2000, num_timesteps=10000000, max_episode_timesteps=500, num_repeat_actions=1,
        # Callback
        callback=episode_finished, callback_episode_frequency=1, callback_timestep_frequency=None,
        # Tqdm
        use_tqdm=True, mean_horizon=100,
        # Evaluation
        evaluation=False, evaluation_callback=None, evaluation_frequency=None,
        max_evaluation_timesteps=None, num_evaluation_iterations=0
    )

    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()