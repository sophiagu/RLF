from gym.envs.registration import register

register(
    id='RLF-v0',
    entry_point='gym_rlf.envs:RLFEnv',
)

register(
    id='MeanReversion-v0',
    entry_point='gym_rlf.envs:MeanReversionEnv',
)

register(
    id='APT-v0',
    entry_point='gym_rlf.envs:APTEnv',
)

register(
    id='DeltaHedging-v0',
    entry_point='gym_rlf.envs:DeltaHedgingEnv',
)