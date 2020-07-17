import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

env = make_vec_env('gym_rlf:MeanReversion-v0', n_envs=1)

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("a2c_mean_reversion")

del model

model = TRPO.load("a2c_mean_reversion")
for episode in range(10):
  obs = env.reset()
  for i in range(1000):
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
  env.render()
env.close()