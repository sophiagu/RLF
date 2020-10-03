# Suppress deprecation warnings from Tensorflow.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
from gym import logger

import argparse
import multiprocessing
import numpy as np
import optuna
import os

from functools import partial
from utils import make_env, ppo2_params
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import ConvexPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines import PPO2

NUM_CPU = multiprocessing.cpu_count()
L = 1000
MAX_PATIENCE = 4
USE_CONVEX_NN = False

def _train(env_id, model_params, total_steps, use_convex_nn=False, is_evaluation=False):
  if is_evaluation: # evaluate_policy() must only take one environment
    envs = SubprocVecEnv([make_env(env_id)])
  else:
    envs = SubprocVecEnv([make_env(env_id) for _ in range(NUM_CPU)])
  envs = VecNormalize(envs) # normalize the envs during training and evaluation

  if use_convex_nn:
    model = PPO2(ConvexPolicy, envs, n_steps=2, nminibatches=1,
                 learning_rate=lambda f: f * 1.0e-5, verbose=1,
                 **model_params)
  else:
    model = PPO2(MlpLstmPolicy, envs, n_steps=2, nminibatches=1,
                 learning_rate=lambda f: f * 1.0e-5, verbose=1,
                 policy_kwargs=dict(act_fun=tf.nn.relu, net_arch=None),
                 **model_params)

  model.learn(total_timesteps=total_steps)
  return envs, model
  
def _search_hparams(env_id, total_steps, trial):
  envs, model = _train(env_id, ppo2_params(trial), total_steps, USE_CONVEX_NN, True)
  mean_reward, _ = evaluate_policy(model, envs, n_eval_episodes=10)
  envs.close()
  # Negate the reward because Optuna minimizes lost.
  return -mean_reward

def _eval_model(model, env_id, L, ob_shape, num_eps, plot=False):
  if num_eps <= 2:
    raise AssertionError('num_eps must be greater than two')

  test_env = SubprocVecEnv([make_env(env_id)])
  sharpe_ratios = []
  for episode in range(num_eps):
    # Padding zeros to the test env to match the shape of the training env.
    zero_completed_obs = np.zeros((NUM_CPU,) + ob_shape)
    zero_completed_obs[0, :] = test_env.reset()
    state = None
    for _ in range(L):
      action, state = model.predict(zero_completed_obs, state=state, deterministic=True)
      zero_completed_obs[0, :], reward, done, _ = test_env.env_method('step', action[0], indices=0)[0]
    sharpe_ratios.append(test_env.env_method('get_sharpe_ratio', indices=0)[0])
    if plot: test_env.env_method('render', indices=0)
  test_env.close()
  
  # Compute the average sharpe ratio after removing the highest and the lowest values.
  sharpe_ratios.sort()
  return sum(sharpe_ratios[1:-1]) / len(sharpe_ratios[1:-1])
  

if __name__ == '__main__':
  logger.set_level(logger.ERROR)

  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str)
  parser.add_argument('--optimize', type=bool, default=False,
                      help='Search for optimal hyperparameters. Drop this flag to run the actual training.')
  parser.add_argument('--num_trials', type=int, default=100,
                      help='Number of trials to search for optimal hyperparameters.')
  parser.add_argument('--num_eps', type=int, default=10,
                      help='Number of episodes to run the final model after training.')
  parser.add_argument('--evaluation_epoches', type=int, default=10,
                      help='The length that the model runs when evaluating hyperparameters.')
  parser.add_argument('--evaluate_model_per_epoches', type=int, default=1,
                      help='How often should we evaluate the model during training.')
  parser.add_argument('--max_train_epoches', type=int, default=10000,
                      help='Max number of epoches that the model runs during training.')
  args = parser.parse_args()
  if args.env == 'mean_reversion':
    env_id = 'gym_rlf:MeanReversion-v0'
    study_name = 'mean-reversion-study'
  elif args.env == 'apt':
    env_id = 'gym_rlf:APT-v0'
    study_name = 'apt-study'
  elif args.env == 'delta_hedging':
    env_id = 'gym_rlf:DeltaHedging-v0'
    study_name = 'delta-hedging-study'
  else:
    raise NotImplementedError

  study = optuna.create_study(
    study_name=study_name,
    storage='sqlite:///{}.db'.format(args.env),
    load_if_exists=True)

  if args.optimize:
    study.optimize(
      partial(_search_hparams, env_id, args.evaluation_steps),
      n_trials=args.num_trials,
      n_jobs=NUM_CPU)
    # df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    # print(df)
    print('best hyperparamters found =', study.best_params)
    print('best value achieved =', study.best_value)
    print('best trial =', study.best_trial)

  assert args.max_train_epoches % args.evaluate_model_per_epoches == 0
  best_sr = None
  best_train_epoches = None
  patience_counter = 0
  for i in range(1, args.max_train_epoches // args.evaluate_model_per_epoches + 1):
    envs, model = _train(env_id, study.best_params, i * (args.evaluate_model_per_epoches * L), USE_CONVEX_NN)
    sharpe_ratio = _eval_model(model, env_id, L, envs.observation_space.shape, 7)
    if best_sr is None or sharpe_ratio > best_sr:
      best_sr = sharpe_ratio
      best_train_epoches = i * args.evaluate_model_per_epoches
      patience_counter = 0
      model.save(args.env)
    else:
      patience_counter += 1
      if patience_counter > MAX_PATIENCE:
        print(
          'Training stopped after {} epoches with the best sharpe ratio {} and the best training epoches {}.'
          .format(i * args.evaluate_model_per_epoches, best_sr, best_train_epoches))
        break
  if i > args.max_train_epoches // args.evaluate_model_per_epoches:
    print(
      'Training stopped with the best sharpe ratio {} and the best training epoches {}.'
      .format(best_sr, best_train_epoches))
  print('best average sharpe ratio =', best_sr)

  del model
  model = PPO2.load(args.env)
  sharpe_ratio = _eval_model(model, env_id, L, envs.observation_space.shape, args.num_eps, True)
  print('average sharpe ratio =', sharpe_ratio)
  envs.close()