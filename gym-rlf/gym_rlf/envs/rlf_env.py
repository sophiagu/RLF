import gym
import numpy as np
import os
import tempfile

from gym_rlf.envs.Parameters import TickSize, LotSize, M, K, p_e

# OpenAI Baselines recommends to normalize continuous action space because the Baselines
# agents only sample actions from a standard Gaussian.
# We use a space normalizer to rescale the action space to [-LotSize * K, LotSize * K].
action_space_normalizer = LotSize * K

MAX_HOLDING = LotSize * M
MIN_PRICE = round(TickSize, 2) # strictly positive price
MAX_PRICE = round(TickSize * 1000, 2)


class RLFEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, episode_size, plot_folder_name=None):
    self.L = episode_size
    self.render_counts = 0
    self.folder_name = None
    if plot_folder_name is not None:
      # Save plots to a newly created folder under plot_folder_name.
      self.folder_name = plot_folder_name + next(tempfile._get_candidate_names())

  def reset(self):
    self.step_counts = 0
    self.prices = np.zeros(self.L+1)
    self.prices[0] = p_e
    self.positions = np.zeros(self.L+1)
    self.rewards = np.zeros(self.L+1)
    self.profits = np.zeros(self.L+1)
    self.costs = np.zeros(self.L+1)

  def get_state(self):
    raise NotImplementedError

  def step(self, action):
    raise NotImplementedError

  def render(self, mode='human'):
    if mode == 'human':
      self.render_counts += 1
      if self.folder_name is not None and not os.path.exists(self.folder_name):
        os.makedirs(self.folder_name)

      print('sharpe ratio =', 16 * np.mean(self.rewards) / np.std(self.rewards))
    else:
      raise NotImplementedError

  def close(self):
    pass
