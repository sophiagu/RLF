import gym
import numpy as np
import os
import tempfile

from gym_rlf.envs.Parameters import TickSize, LotSize, M, K, p_e

# Stable Baselines recommends to normalize continuous action space because the Baselines
# agents only sample actions from a standard Gaussian.
# We use a space normalizer to rescale the action space to [-LotSize * K, LotSize * K].
action_space_normalizer = LotSize * K

MAX_HOLDING = LotSize * M
MIN_PRICE = round(TickSize, 2) # strictly positive price
MAX_PRICE = round(TickSize * 10000, 2)


class RLFEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, plot_folder_name=None):
    self._L = 1000 # number of time steps in each episode
    self._render_counts = 0
    self._folder_name = None
    if plot_folder_name is not None:
      # Save plots to a newly created folder under plot_folder_name.
      self._folder_name = plot_folder_name + next(tempfile._get_candidate_names())

  def reset(self):
    self._step_counts = 1
    self._prices = np.zeros(self._L)
    self._prices[0] = self._prices[1] = p_e
    self._positions = np.zeros(self._L)
    self._pnls = np.zeros(self._L)
    self._costs = np.zeros(self._L)
    self._states = []
    self._actions = []

  def get_sharpe_ratio(self):
    # sharpe ratio of the annualized PnL
    return 16 * np.mean(self._pnls) / np.std(self._pnls)

  def _get_state(self):
    raise NotImplementedError

  def _learn_func_property(self):
    # Returns the fraction of the number of states that violate a function property.
    # The return value is always between 0 and 1.
    pass

  def step(self, action):
    raise NotImplementedError

  def render(self, mode='human'):
    if mode == 'human':
      self._render_counts += 1
      if self._folder_name is not None and not os.path.exists(self._folder_name):
        os.makedirs(self._folder_name)

      print('sharpe ratio =', self.get_sharpe_ratio())
    else:
      raise NotImplementedError

  def close(self):
    pass
