from gym.envs.registration import register

register(
    id='mr-v0',
    entry_point='gym_mr.envs:MrEnv',
)
