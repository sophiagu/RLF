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
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, SigmoidMlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines import PPO2

NUM_CPU = multiprocessing.cpu_count()
L = 1000
MAX_PATIENCE = 5

def _train(env_id, model_params, total_epochs, use_sigmoid_layer=False, is_evaluation=False):
  if is_evaluation: # evaluate_policy() must only take one environment
    envs = SubprocVecEnv([make_env(env_id)])
  else:
    envs = SubprocVecEnv([make_env(env_id) for _ in range(NUM_CPU)])
  envs = VecNormalize(envs) # normalize the envs during training and evaluation

  # activation fn: use tanh for delta hedging and relu for mean reversion
  # learning rate: use 1e-7 for delta hedging and 1e-5 for mean reversion
  if use_sigmoid_layer:
    model = PPO2(SigmoidMlpPolicy, envs, n_steps=1, nminibatches=1,
                 learning_rate=lambda f: f * 1e-7, verbose=1,
                 policy_kwargs=dict(act_fun=tf.nn.tanh),
                 **model_params)
  else:
    model = PPO2(MlpLstmPolicy, envs, n_steps=1, nminibatches=1,
                 learning_rate=lambda f: f * 1e-7, verbose=1,
                 policy_kwargs=dict(act_fun=tf.nn.tanh),
                 **model_params)

  model.learn(total_timesteps=total_epochs * L)
  return envs, model
  
def _search_hparams(env_id, total_epochs, use_sigmoid_layer, trial):
  envs, model = _train(env_id, ppo2_params(trial), total_epochs, use_sigmoid_layer, True)
  mean_reward, _ = evaluate_policy(model, envs, n_eval_episodes=10)
  envs.close()
  # Negate the reward because Optuna minimizes lost.
  return -mean_reward

def _eval_model(model, env_id, ob_shape, num_eps, plot=False):
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
  
  # Return the average sharpe ratio
  return sum(sharpe_ratios) / len(sharpe_ratios)
  

if __name__ == '__main__':
  logger.set_level(logger.ERROR)

  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str)
  parser.add_argument('--use_sigmoid_layer', type=bool, default=False,
                      help='Whether or not to use SigmoidMlpPolicy. Drop this flag to use MlpPolicy.')
  parser.add_argument('--optimize', type=bool, default=False,
                      help='Search for optimal hyperparameters. Drop this flag to run the actual training.')
  parser.add_argument('--num_trials', type=int, default=10,
                      help='Number of trials to search for optimal hyperparameters.')
  parser.add_argument('--evaluation_epochs', type=int, default=10,
                      help='The length that the model runs when evaluating hyperparameters.')
  parser.add_argument('--evaluate_model_per_epochs', type=int, default=10,
                      help='How often should we evaluate the model during training.')
  parser.add_argument('--max_train_epochs', type=int, default=1000,
                      help='Max number of epochs that the model runs during training.')
  parser.add_argument('--num_random_initializations', type=int, default=10,
                      help='Number of trials for different initializations of weights.')
  parser.add_argument('--num_eps', type=int, default=10,
                      help='Number of episodes to run the final model after training.')
  args = parser.parse_args()

  ######## Setup ########
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

  ######## Search for hyperparameters ########
  if args.optimize:
    study.optimize(
      partial(_search_hparams, env_id, args.evaluation_epochs, args.use_sigmoid_layer),
      n_trials=args.num_trials,
      n_jobs=NUM_CPU)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    num_rows = len(df.index)
    print(df.tail(min(args.num_trials, num_rows)))
    print('best hyperparamters found =', study.best_params)
    print('best value achieved =', -study.best_value)
    print('best trial =', study.best_trial)

  ######### Training ########
  assert args.max_train_epochs % args.evaluate_model_per_epochs == 0
  best_sr = None
  best_train_epochs = None

  for _ in range(args.num_random_initializations):
    patience_counter = 0
    for i in range(1, args.max_train_epochs // args.evaluate_model_per_epochs + 1):
      envs, model = _train(
        env_id,
        study.best_params,
        args.evaluate_model_per_epochs * i,
        args.use_sigmoid_layer)

      sharpe_ratio = _eval_model(model, env_id, envs.observation_space.shape, 10)
      if best_sr is None or sharpe_ratio > best_sr:
        best_sr = sharpe_ratio
        best_train_epochs = args.evaluate_model_per_epochs * i
        patience_counter = 0
        model.save(args.env)
      else:
        patience_counter += 1
        if patience_counter > MAX_PATIENCE: break

  print('best average training sharpe ratio={}; best training epochs={}'.format(best_sr, best_train_epochs))
  del model

  ######## Testing ########
  model = PPO2.load(args.env)
  sharpe_ratio = _eval_model(model, env_id, envs.observation_space.shape, args.num_eps, True)
  print('average test sharpe ratio={}'.format(sharpe_ratio))
  print('hyperparamters used={}'.format(study.best_params))
