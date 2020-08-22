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
    self._L = episode_size
    self._render_counts = 0
    self._folder_name = None
    if plot_folder_name is not None:
      # Save plots to a newly created folder under plot_folder_name.
      self._folder_name = plot_folder_name + next(tempfile._get_candidate_names())

  def reset(self):
    self._step_counts = 0
    self._prices = np.zeros(self._L+1)
    self._prices[0] = p_e
    self._positions = np.zeros(self._L+1)
    self._rewards = np.zeros(self._L+1)
    self._profits = np.zeros(self._L+1)
    self._costs = np.zeros(self._L+1)
    self._states = []
    self._actions = []

  def _get_state(self):
    raise NotImplementedError
    
  def get_sharpe_ratio(self):
    return 16 * np.mean(self._rewards) / np.std(self._rewards)

  def _learn_func_property(self, func):
    num_past_data = len(self._states) - 1
    if num_past_data <= 0: return 0

    penalty = 0
    for i in range(num_past_data):
      penalty += func(self._states[i], self._states[-1], self._actions[i], self._actions[-1])
    return penalty / num_past_data
    
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
