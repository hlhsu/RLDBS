from gym.envs.registration import register

register(
    id='oscillator-v0',
    entry_point='gym_oscillator.envs:oscillatorEnv',
)

register(
    id='oscillator-v1',
    entry_point='gym_oscillator.envs:oscillatorEnv_Rec',
)
register(
    id='oscillator-v2',
    entry_point='gym_oscillator.envs:oscillatorEnv_Bal',
)


register(
    id='oscillator-v3',
    entry_point='gym_oscillator.envs:oscillatorEnv_dis',
)
