from gym.envs.registration import register

register(
    id='achtung-v0',
    entry_point='gym_achtung.envs:AchtungEnv'
)
