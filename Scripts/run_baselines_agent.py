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
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines import PPO2

NUM_CPU = multiprocessing.cpu_count()
L = 1000
MAX_PATIENCE = 10

def _train(env_id, model_params, total_steps, is_evaluation=False):
  if is_evaluation: # evaluate_policy() must only take one environment
    envs = SubprocVecEnv([make_env(env_id)])
  else:
    envs = SubprocVecEnv([make_env(env_id) for _ in range(NUM_CPU)])
  envs = VecNormalize(envs) # normalize the envs during training and evaluation

  # Load pretrained model during training.
  if not is_evaluation and os.path.exists(env_id):
    model = PPO2.load(env_id)
  else:
    model = PPO2(MlpLstmPolicy, envs, n_steps=8, nminibatches=1,
                 learning_rate=lambda f: f * .001, verbose=1,
                 policy_kwargs=dict(act_fun=tf.nn.relu, net_arch=None),
                 **model_params)

  model.learn(total_timesteps=total_steps)
  return envs, model
  
def _search_hparams(env_id, total_steps, trial):
  envs, model = _train(env_id, ppo2_params(trial), total_steps, True)
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
  parser.add_argument('--evaluation_steps', type=int, default=500000,
                      help=('Number of total timesteps that the model runs when evaluating hyperparameters.'
                            'This number must be a multiple of the environment episode size L.'))
  parser.add_argument('--max_train_steps', type=int, default=5000000,
                      help=('Max number of total timesteps that the model runs during training.'
                            'This number must be a multiple of the environment episode size L.'))
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

  # Evaluate the model every 100 x L timesteps.
  # Stop when the evaluation result drops by MAX_PATIENCE number of times.
  assert args.max_train_steps % (100 * L) == 0
  best_sr = 0
  patience_counter = 0
  for i in range(args.max_train_steps // (100 * L)):
    envs, model = _train(env_id, study.best_params, 100 * L)
    sharpe_ratio = _eval_model(model, env_id, L, envs.observation_space.shape, 7)
    if sharpe_ratio > best_sr:
      sharpe_ratio = best_sr
      patience_counter = 0
      model.save(args.env)
    elif best_sr >= 1:
      patience_counter += 1
      if patience_counter > MAX_PATIENCE:
        print('Training stopped after {} episodes with sharpe ratio {}.'.format(i + 1, best_sr))
        break
  print('best average sharpe ratio =', best_sr)
  if not os.path.exists(args.env):
    print('Training finished without finding a good policy!')
    model.save(args.env)

  del model
  model = PPO2.load(args.env)
  sharpe_ratio = _eval_model(model, env_id, L, envs.observation_space.shape, args.num_eps, True)
  print('average sharpe ratio =', sharpe_ratio)
  envs.close()