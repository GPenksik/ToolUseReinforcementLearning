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
import time

sys.path.append(os.path.abspath("/home/adf/exp715/EnvTest/Lib/site-packages/"))

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

warnings.filterwarnings("ignore", category=DeprecationWarning)


# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000

# python examples/openai_gym.py CartPole-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 2000 -m 200


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('run_number', help="Consecutive number of this run")
    parser.add_argument('gym_id', help="Id of the Gym environment")
    # parser.add_argument('-i', '--import-modules', help="Import module(s) required for environment")
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

    parser.add_argument('-rp', '--repeat', type=int, default=6, help='How many times to repeat an action')

    parser.add_argument('-bt', '--batch-size', type=int)
    parser.add_argument('-mc', '--memory-capacity', type=int)
    parser.add_argument('-os', '--optimization-steps', type=int)
    parser.add_argument('-bs', '--baseline-steps', type=int)

    # parser.add_argument('--monitor', help="Save results to this directory")
    # parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    # parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")
    parser.add_argument('-te', '--test', action='store_true', default=False, help="Test agent without learning.")
    parser.add_argument('-sd', '--seed', type=int, default=None, help='Random seed for this trial')
    # parser.add_argument('-sl', '--sleep', type=float, default=None, help="Slow down simulation by sleeping for x seconds (fractions allowed).")
    # parser.add_argument('--job', type=str, default=None, help="For distributed mode: The job type of this agent.")
    # parser.add_argument('--task', type=int, default=0, help="For distributed mode: The task index of this agent.")

    args = parser.parse_args()

    # SET BASIC PARAMETERS

    random_seed = args.seed
    agent_save_period = args.save_episodes
    visualize_period = 10
    run_number = args.run_number


    load_agent = False
    if args.load:
        load_agent = True
        agent_filename = args.load
    else:
        load_agent = False

    if args.visualize:
        to_visualize = True
    else:
        to_visualize = False


    # Set logging level
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # if args.import_modules is not None:
    #    for module in args.import_modules.split(','):
    #        importlib.import_module(name=module)

    environment = OpenAIGym(
        gym_id=args.gym_id,
        monitor='/home/adf/exp715/Box2dEnv/Box2dEnv/savemonitor/',
        monitor_safe=False,
        monitor_video=0,
        visualize=False
        # True to visualize first run. Otherwise visualisation is set in episode_finished() method
    )

    # Set random seed, reward level and random level for environment
    environment.gym.seed(random_seed)
    environment.gym.unwrapped.set_reward(args.reward_level)
    environment.gym.unwrapped.set_random(args.random_level)
    environment.gym.unwrapped.set_reward_scale(args.reward_scale)
    # Initialize Agent-Network-Model objects

    # with open('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\examples\\configs\\ppo.json', 'r') as fp:
    #     agent = json.load(fp=fp)
    with open('/home/adf/exp715/Box2dEnv/examples/configs/{}'.format(args.agent), 'r') as fp:
        agentSpec = json.load(fp=fp)

    # with open('C:\\Users\\genia\\Source\\Repos\\Box2dEnv\\examples\\configs\\mlp2_network.json', 'r') as fp:
    #     network = json.load(fp=fp)
    with open('/home/adf/exp715/Box2dEnv/examples/configs/{}'.format(args.network), 'r') as fp:
        network = json.load(fp=fp)

    # Set agent hyper parameters
    # agentSpec['memory'].update(capacity=args.memory_capacity)
    agentSpec.update(optimization_steps=args.optimization_steps)
    agentSpec['baseline_optimizer'].update(num_steps=args.baseline_steps)
    agentSpec['update_mode'].update(batch_size=args.batch_size)
    agentSpec['update_mode'].update(frequency=args.batch_size)
    agentSpec['baseline']['sizes'] = [args.network_size,args.network_size]
    network[0].update(size=args.network_size)
    network[1].update(size=args.network_size)

    agent = Agent.from_spec(
        spec=agentSpec,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network,
            random_seed=random_seed
        )
    )

    print("Agent memory ",agent.memory['capacity'])
    print("Agent baseline steps", agent.baseline_optimizer['num_steps'])
    print("Agent optimizer steps", agent.optimizer['num_steps'])

    if load_agent:
        agent.restore_model(directory='/home/adf/exp715/Box2dEnv/Box2dEnv/saves/modelSave', file=agent_filename)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=args.repeat
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

        if r.episode % visualize_period == 0:
            if to_visualize:
                environment.visualize = True  # Set to true to visualize
        else:
            environment.visualize = False

        save_period = 20
        if r.episode % save_period == 0:
            with open('/home/adf/exp715/Box2dEnv/Box2dEnv/saves/{}.csv'.format(nName), 'a+') as csv:
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
                    '/home/adf/exp715/Box2dEnv/Box2dEnv/saves/{}'.format(nName), r.episode))
            r.agent.save_model(directory='/home/adf/exp715/Box2dEnv/Box2dEnv/saves/modelSave/{}-{}'.format(nName, r.episode),
                               append_timestep=False)

        return True

    runner.run(
        num_timesteps=args.timesteps,
        num_episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished,
        testing=args.test,
        sleep=None
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))
    logger.info("Time taken = {}".format( time.time()-start_time))

if __name__ == '__main__':
    main()
