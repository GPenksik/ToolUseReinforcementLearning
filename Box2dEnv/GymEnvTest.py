import sys
import os
sys.path.append(os.path.abspath("C:\\Users\\genia\\source\\repos\\Box2dEnv"))

import gym
#import env_test
#envlander = gym.make('LunarLander-v2')
env = gym.make('EnvTestContinuousR-v2')
env.unwrapped.seed(20)
env.unwrapped.set_repeat(12)
env.unwrapped.set_task(1)
env.unwrapped.set_random(1)

#env.seed(1)
env.reset()

env.render()
done = False
actionval = 1
counter = 0
cumulativeReward = 0

while done == False:
    #action = [0.9*actionval,-actionval,0]
    action = [actionval, -actionval, 0]

    #action = [0,0,0]
    if counter > 300/12:
        action = [-actionval, 0, 0]
    if counter > 570/12:
        action = [0, 0.6*actionval, -0.2]
    if counter > 900/12:
        action = [0, 0, 0]
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