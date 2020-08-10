import gym
from gym import wrappers, logger

import argparse
import numpy as np
import pickle
import json, sys, os
from os import path
from _policies import BinaryActionLinearPolicy, ContinuousActionLinearPolicy # Different file so it can be unpickled

ENV_ID = 'gym_rlf:MeanReversion-v0'


class RandomAgent:
  def __init__(self):
    self.env = gym.make(ENV_ID)
    self.env.reset()

  def act(self, observation, reward, done):
    return self.env.action_space.sample()


class CEMAgent:
  def __init__(self, args):
    env = gym.make(ENV_ID)
    np.random.seed(0)
    self.params = dict(n_iter=500, batch_size=25, elite_frac=.1)
    self.num_steps = 1000

    self.outdir = 'cem-agent-results'
    self.env = wrappers.Monitor(env, self.outdir, force=True)
    self.env.reset()
    self.args = args
    self.n_in = self.env.observation_space.shape[0]
    self.n_out = self.env.action_space.shape[0]
    
    self.agent = ContinuousActionLinearPolicy(np.zeros((self.n_in+1)*self.n_out), self.n_in, self.n_out)
    # Train the agent, and snapshot each stage.
    for (i, iterdata) in enumerate(
      self.cem(self.noisy_evaluation, np.zeros((self.n_in+1)*self.n_out), **self.params)):
      print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
      self.agent = ContinuousActionLinearPolicy(iterdata['theta_mean'], self.n_in, self.n_out)
      if args.display: do_rollout(self.agent, env, self.num_steps, render=True)
      self.writefile('agent-%.4i.pkl'%i, str(pickle.dumps(self.agent, -1)))

    # Write out the env at the end so we store the parameters of this environment.
    info = {}
    info['params'] = self.params
    info['argv'] = sys.argv
    info['env_id'] = self.env.spec.id
    self.writefile('info.json', json.dumps(info))
    
  def writefile(self, fname, s):
    with open(path.join(self.outdir, fname), 'w') as fh: fh.write(s)
      
  def act(self, observation, reward, done):
    return self.agent.act(observation)
      
  def noisy_evaluation(self, theta):
    agent = ContinuousActionLinearPolicy(theta, self.n_in, self.n_out)
    rew, T = self.do_rollout(agent, self.env, self.num_steps)
    return rew

  def cem(self, f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function
    Args:
      f: a function mapping from vector -> scalar
      th_mean (np.array): initial mean over input distribution
      batch_size (int): number of samples of theta to evaluate per batch
      n_iter (int): number of batches
      elite_frac (float): each batch, select this fraction of the top-performing samples
      initial_std (float): initial standard deviation over parameter vectors
    returns:
      A generator of dicts. Subsequent dicts correspond to iterations of CEM algorithm.
      The dicts contain the following values:
      'ys' :  numpy array with values of function evaluated at current population
      'ys_mean': mean value of function over current population
      'theta_mean': mean value of the parameter vector over current population
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
      ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
      ys = np.array([f(th) for th in ths])
      elite_inds = ys.argsort()[::-1][:n_elite]
      while ys[elite_inds].any() < 0:
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
      elite_ths = ths[elite_inds]
      th_mean = elite_ths.mean(axis=0)
      th_std = elite_ths.std(axis=0)
      yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

  def do_rollout(self, agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
      a = agent.act(ob)
      (ob, reward, done, _info) = env.step(a)
      total_rew += reward
      if render and t%3==0: env.render()
      if done: break
    return total_rew, t+1


class SimpleTestAgent:
  def __init__(self):
    self.env = gym.make(ENV_ID)
    self.env.reset()
    
    print('observation space:', self.env.observation_space)
    print('action space:', self.env.action_space)

  def act(self, observation, reward, done):
    if observation[1] <= 30:
      return 1
    elif observation[1] <= 40:
      return .5
    elif observation[1] >= 70:
      return -1
    elif observation[1] >= 60:
      return -.5
    else:
      return 0

  
if __name__ == '__main__':
  logger.set_level(logger.ERROR)
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--agent')
  parser.add_argument('--display', action='store_true')
  args = parser.parse_args()
  
  if args.agent == 'random':
    agent = RandomAgent()
  elif args.agent == 'cem':
    agent = CEMAgent(args)
  elif args.agent == 'test':
    agent = SimpleTestAgent()
  else:
    raise NotImplementedError

  episode_count = 10
  total_reward = 0
  done = False
  for i in range(1, episode_count+1):
    ob = agent.env.reset()
    t = 0
    while not done:
      t += 1
      action = agent.act(ob, reward, done)
      ob, reward, done, _ = agent.env.step(action)
      total_reward += reward
    print('Episode {} terminated after {} steps with total reward = {}'.format(i, t, total_reward))
    total_reward = 0
    agent.env.render()

  agent.env.close()