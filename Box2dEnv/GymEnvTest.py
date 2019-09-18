import sys
import os
sys.path.append(os.path.abspath("C:\\Users\\genia\\source\\repos\\Box2dEnv"))

import gym
#import env_test
#envlander = gym.make('LunarLander-v2')
env = gym.make('EnvTestContinuousR-v2')

#env.seed(1)
env.reset()

env.render()
done = False
actionval = 1
counter = 0
cumulativeReward = 0

while done == False:
    #action = [0.9*actionval,-actionval,0]
    action = [0.0,-actionval,0]

    #action = [0,0,0]
    if counter > 280:
        action = [-actionval,0,0]
    if counter > 450:
        action = [0,actionval,-0.2]
    if counter > 580:
        action = [0,0,0]
    #action = [0,-0.1,0]
    state, reward, done, [] = env.step(action)
    a = env.render()
    counter += 1
    cumulativeReward += reward
    #if counter > 30:
    #    counter = 0
    #    action += 1

    #if action > 3:
    #    action = 0
    #print(reward)
    #print(reward)
print(cumulativeReward)
env.close()