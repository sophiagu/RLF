import gym

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C

env = gym.make('gym_rlf:MeanReversion-v0')
check_env(env)

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=500000)
model.save("a2c_mean_reversion")

del model

model = A2C.load("a2c_mean_reversion")
for episode in range(10):
  obs = env.reset()
  for i in range(1000):
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
  env.render()
env.close()