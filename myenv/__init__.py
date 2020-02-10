from gym.envs.registration import register

register(
    id='MyDampAcrobot-v0',
    entry_point='myenv.acrobot_damp:AcrobotEnv',
)

register(
    id='MyDampCartPole-v0',
    entry_point='myenv.cartpole_damp:CartPoleEnv',
)

register(
    id='MyDampPendulum-v0',
    entry_point='myenv.pendulum_damp:PendulumEnv',
)

register(
    id='My_FA_Acrobot-v0',
    entry_point='myenv.fa_acrobot:AcrobotEnv',
)

register(
    id='My_FA_CartPole-v0',
    entry_point='myenv.fa_cartpole:CartPoleEnv',
)

register(
    id='MyCartPole-v0',
    entry_point='myenv.cartpole:CartPoleEnv',
)

register(
    id='MyAcrobot-v0',
    entry_point='myenv.acrobot:AcrobotEnv',
)

register(
    id='MyPendulum-v0',
    entry_point='myenv.pendulum:PendulumEnv',
)
