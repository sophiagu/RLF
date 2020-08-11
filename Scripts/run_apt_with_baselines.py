# Suppress deprecation warnings from Tensorflow.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
from gym import logger

import argparse
import multiprocessing
import numpy as np
import optuna

from functools import partial
from utils import make_env, ppo2_params, a2c_params
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines import A2C, PPO2

ENV_ID = 'gym_rlf:APT-v0'
NUM_CPU = multiprocessing.cpu_count()
L = 100

def train(agent, model_params, is_evaluation):
  if is_evaluation: # evaluate_policy() must only take one environment
    envs = SubprocVecEnv([make_env(ENV_ID)])
  else:
    envs = SubprocVecEnv([make_env(ENV_ID) for _ in range(NUM_CPU)])
  envs = VecNormalize(envs) # normalize the envs during training and evaluation

  if agent == 'ppo2':
    model = PPO2(MlpLstmPolicy, envs, nminibatches=1, verbose=1, **model_params)
  elif agent == 'a2c':
    model = A2C(MlpLstmPolicy, envs, verbose=1, **model_params)

  # NOTE: Training or evaluation steps must be multiples of test size L.
  if is_evaluation:
    model.learn(total_timesteps=50000)
  else:
    model.learn(total_timesteps=100000)

  envs.close()
  return model
  
def search_hparams(agent, trial):
  if agent == 'ppo2':
    model_params = ppo2_params(trial)
  elif agent == 'a2c':
    model_params = a2c_params(trial)

  model = train(agent, model_params, True)
  eval_env = SubprocVecEnv([make_env(ENV_ID)])
  eval_env = VecNormalize(eval_env)
  mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
  
  eval_env.close()
  # Negate the reward because Optuna maximizes the negative log likelihood.
  return -mean_reward


if __name__ == '__main__':
  logger.set_level(logger.ERROR)

  parser = argparse.ArgumentParser()
  parser.add_argument('--agent', type=str, default='ppo2')
  parser.add_argument('--optimize', type=bool, default=False)
  args = parser.parse_args()
  agent = args.agent.lower()
  if agent not in ['ppo2', 'a2c']:
    raise NotImplementedError

  study = optuna.create_study(
    study_name=agent + '-apt-study',
    storage='sqlite:///{}_apt.db'.format(agent),
    load_if_exists=True)

  if args.optimize:
    study.optimize(partial(search_hparams, agent) , n_trials=100, n_jobs=NUM_CPU)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)
    print('best hyperparamters found =', study.best_params)
    print('best value achieved =', study.best_value)
    print('best trial =', study.best_trial)

  model = train(agent, study.best_params, False)
  model.save(agent + '_apt')

  del model
  if agent == 'ppo2':
    model = PPO2.load(agent + '_apt')
  elif agent == 'a2c':
    model = A2C.load(agent + '_apt')

  test_env = SubprocVecEnv([make_env(ENV_ID)])
  for episode in range(10):
    # Padding zeros to the test env to match the shape of the training env.
    # The 2nd entry is the shape of the observation space.
    zero_completed_obs = np.zeros((NUM_CPU,) + (3,))
    zero_completed_obs[0, :] = test_env.reset()
    state = None
    for _ in range(L):
      action, state = model.predict(zero_completed_obs, state=state, deterministic=True)
      zero_completed_obs[0, :], reward, done, _ = test_env.env_method('step', action[0], indices=0)[0]
    test_env.env_method('render', indices=0)

  test_env.close()