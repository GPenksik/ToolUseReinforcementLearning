from gym.envs.registration import register

print("init register")

register(
    id='EnvTest-v0',
    entry_point='env_test.envs:EnvTest',
)