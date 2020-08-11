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

NUM_CPU = multiprocessing.cpu_count()

def train(env_id, agent, model_params, is_evaluation, evaluation_steps, train_steps):
  if is_evaluation: # evaluate_policy() must only take one environment
    envs = SubprocVecEnv([make_env(env_id)])
  else:
    envs = SubprocVecEnv([make_env(env_id) for _ in range(NUM_CPU)])
  envs = VecNormalize(envs) # normalize the envs during training and evaluation

  if agent == 'ppo2':
    model = PPO2(MlpLstmPolicy, envs, nminibatches=1, verbose=1, **model_params)
  elif agent == 'a2c':
    model = A2C(MlpLstmPolicy, envs, verbose=1, **model_params)

  if is_evaluation:
    model.learn(total_timesteps=evaluation_steps)
  else:
    model.learn(total_timesteps=train_steps)

  return envs, model
  
def search_hparams(env_id, agent, evaluation_steps, train_steps, trial):
  if agent == 'ppo2':
    model_params = ppo2_params(trial)
  elif agent == 'a2c':
    model_params = a2c_params(trial)

  envs, model = train(env_id, agent, model_params, True, evaluation_steps, train_steps)
  mean_reward, _ = evaluate_policy(model, envs, n_eval_episodes=10)
  
  envs.close()
  # Negate the reward because Optuna maximizes the negative log likelihood.
  return -mean_reward


if __name__ == '__main__':
  logger.set_level(logger.ERROR)

  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str)
  parser.add_argument('--agent', type=str, default='ppo2')
  parser.add_argument('--optimize', type=bool, default=True,
                      help='Whether to search for optimal hyperparameters.')
  parser.add_argument('--num_trials', type=int, default=100,
                      help='Number of trials to search for optimal hyperparameters.')
  parser.add_argument('--num_eps', type=int, default=10,
                      help='Number of episodes to run the final model after training.')
  parser.add_argument('--evaluation_steps', type=int, default=500000,
                      help=('Number of total timesteps that the model runs when evaluating hyperparameters.'
                            'This number must be a multiple of the environment episode size L.'))
  parser.add_argument('--train_steps', type=int, default=1000000,
                      help=('Number of total timesteps that the model runs during training.'
                            'This number must be a multiple of the environment episode size L.'))
  args = parser.parse_args()
  agent = args.agent.lower()
  if agent not in ['ppo2', 'a2c']:
    raise NotImplementedError

  if args.env == 'mean_reversion':
    env_id = 'gym_rlf:MeanReversion-v0'
    study_name = agent + '-mean-reversion-study'
    L = 1000
  elif args.env == 'apt':
    env_id = 'gym_rlf:APT-v0'
    study_name = agent + '-apt-study'
    L = 100
  else:
    raise NotImplementedError
  study = optuna.create_study(
    study_name=study_name,
    storage='sqlite:///{}_{}.db'.format(agent, args.env),
    load_if_exists=True)

  if args.optimize:
    study.optimize(
      partial(search_hparams, env_id, agent, args.evaluation_steps, args.train_steps),
      n_trials=args.num_trials,
      n_jobs=NUM_CPU)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)
    print('best hyperparamters found =', study.best_params)
    print('best value achieved =', study.best_value)
    print('best trial =', study.best_trial)

  envs, model = train(env_id, agent, study.best_params, False, args.evaluation_steps, args.train_steps)
  model.save(agent + '_' + args.env)

  del model
  if agent == 'ppo2':
    model = PPO2.load(agent + '_' + args.env)
  elif agent == 'a2c':
    model = A2C.load(agent + '_' + args.env)

  test_env = SubprocVecEnv([make_env(env_id)])
  for episode in range(args.num_eps):
    # Padding zeros to the test env to match the shape of the training env.
    zero_completed_obs = np.zeros((NUM_CPU,) + envs.observation_space.shape)
    zero_completed_obs[0, :] = test_env.reset()
    state = None
    for _ in range(L):
      action, state = model.predict(zero_completed_obs, state=state, deterministic=True)
      zero_completed_obs[0, :], reward, done, _ = test_env.env_method('step', action[0], indices=0)[0]
    test_env.env_method('render', indices=0)

  envs.close()
  test_env.close()