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
from tensorforce.contrib.openai_gym import OpenAIGym

sys.path.append(os.path.abspath('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\'))
sys.path.append(os.path.abspath('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\examples'))


warnings.filterwarnings("ignore", category=DeprecationWarning)


# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000

# python examples/openai_gym.py CartPole-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 2000 -m 200


def main():

    # SET BASIC PARAMETERS
    start_time = time.time()
    random_seed = 20
    agent_save_period = 500
    visualize_period = 1
    run_number = 999

    load_agent = False
    agent_filename = '371-P33-27-PPO-2000'
    to_visualize = True

    # Set logging level
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # if args.import_modules is not None:
    #    for module in args.import_modules.split(','):
    #        importlib.import_module(name=module)

    environment = OpenAIGym(
        # gym_id="BlockPushSimpleContinuous-v2",
        gym_id="EnvTestContinuousR-v2",
        monitor='C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\savemonitor\\',
        monitor_safe=False,
        monitor_video=0,
        visualize=to_visualize
        # True to visualize first run. Otherwise visualisation is set in episode_finished() method
    )

    # Set random seed for environment
    environment.gym.seed(random_seed)
    environment.gym.unwrapped.set_reward(2)
    environment.gym.unwrapped.set_random(3)
    environment.gym.unwrapped.set_reward_scale(6)

    # Initialize Agent-Network-Model objects

    with open('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\examples\\configs\\300-ppo.json', 'r') as fp:
        agentSpec = json.load(fp=fp)

    with open('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\examples\\configs\\300-mlp2_network.json', 'r') as fp:
        network = json.load(fp=fp)

    #agentSpec['update_mode'].update(batch_size=24)
    #agentSpec['update_mode'].update(frequency=24)
    #agentSpec['baseline']['sizes'] = [512,512]
    #network[0].update(size=512)
    #network[1].update(size=512)

    agent = Agent.from_spec(
        spec=agentSpec,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network,
            random_seed=random_seed
        )
    )

    print("Agent memory ", agent.memory['capacity'])
    print("Agent baseline steps", agent.baseline_optimizer['num_steps'])
    print("Agent optimizer steps", agent.optimizer['num_steps'])

    if load_agent:
        agent.restore_model(directory='C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\modelSave', file=agent_filename)

    # if args.load:
    #    load_dir = os.path.dirname(args.load)
    #    if not os.path.isdir(load_dir):
    #        raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
    #    agent.restore_model(args.load)

    # to_save = False
    # if to_save:
    #     save_dir = os.path.dirname('\\saves\\')
    #     if not os.path.isdir(save_dir):
    #         try:
    #             os.mkdir(save_dir, 0o755)
    #         except OSError:
    #             raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    debug = False
    if debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=12
    )

    report_frequently = True
    if report_frequently:
        report_episodes = 1
    else:
        report_episodes = 100

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    # Naming variables
    nNum = str(run_number).zfill(3)
    task = environment.gym.unwrapped.task
    if task == 'LIFT':
        nTask = 'L'
    else:
        nTask = 'P'
    nReward = environment.gym.unwrapped.reward_level
    nRandom = environment.gym.unwrapped.rand_level
    nSeed = str(random_seed).zfill(2)
    nAlg = 'PPO'

    nName = ("{}-{}{}{}-{}-{}".format(nNum, nTask, nReward, nRandom, nSeed, nAlg))

    def episode_finished(r, id_):

        # if r.episode == 1:
        # r.agent.restore_model('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\modelSave')

        save_period = 5
        if r.episode % visualize_period == 0:
            if to_visualize:
                environment.visualize = True  # Set to true to visualize
        else:
            environment.visualize = False

        if r.episode % save_period == 0:
            with open('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\{}.csv'.format(nName), 'a+') as csv:
                for reward in r.episode_rewards[-save_period:]:
                    csv.write("{:2.2f}\n".format(reward))
                # print("Saving, yo!")

        if r.episode % report_episodes == 0:
            # steps_per_second = r.timestep / (time.time() - r.start_time)
            # logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
            #    r.agent.episode, r.episode_timestep, steps_per_second
            # ))
            # logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            # logger.info("Average of last 500 rewards: {:0.2f}".
            #            format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            # logger.info("Average of last 100 rewards: {:0.2f}".
            #            format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
            logger.info("{:6d},    {:+6.2f},     {:+6.2f}".format(r.agent.episode, r.episode_rewards[-1],
                                                                  sum(r.episode_rewards[-100:]) / min(100, len(
                                                                      r.episode_rewards))))

        if r.episode == 1 or (r.episode % agent_save_period == 0):
            logger.info(
                "Saving agent to {} at episode {}".format(
                    'C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\{}'.format(nName), r.episode))
            r.agent.save_model(directory='C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\Box2dEnv\\saves\\modelSave\\{}-{}'.format(nName, r.episode),
                               append_timestep=False)

        return True

    runner.run(
        num_timesteps=20000000,
        num_episodes=10000,
        max_episode_timesteps=1000,
        deterministic=False,
        episode_finished=episode_finished,
        testing=False,
        sleep=None
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
