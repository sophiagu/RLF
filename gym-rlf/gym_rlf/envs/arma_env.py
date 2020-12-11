import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gym import spaces
from gym_rlf.envs.fn_properties import monotonic_decreasing
from gym_rlf.envs.mean_reversion_env import MeanReversionEnv
from gym_rlf.envs.rlf_env import MIN_PRICE, MAX_PRICE
from gym_rlf.envs.Parameters import LotSize, TickSize, Lambda, theta, sigma, kappa, p_e, M, K

# Stable Baselines recommends to normalize continuous action space because the Baselines
# agents only sample actions from a standard Gaussian.
# We use a space normalizer to rescale the action space to [-LotSize * K, LotSize * K].
ACTION_SPACE_NORMALIZER = LotSize * K
MAX_HOLDING = LotSize * M
FUNC_PROPERTY_PENALTY = False


class ARMAEnv(MeanReversionEnv):
  def __init__(self):
    super(ARMAEnv, self).__init__('arma_plots/')

  def _next_price(self, p, p2):
    x = np.log(p / p_e)
    x2 = np.log(p2 / p_e)
    self._rns.append(np.random.normal(0, 1., 1)[0])
    x = x - Lambda * x - Lambda * 2 / 3 * x2 + sigma * self._rns[-1] + theta * sigma * self._rns[-2]
    p_new = p_e * np.exp(x)
    p_new = min(p_new, MAX_PRICE)
    p_new = max(p_new, MIN_PRICE)
    return p_new

  def reset(self):
    super(ARMAEnv, self).reset()

    self._rns = [0]
    return self._get_state()
    
  def _learn_func_property(self):
    if len(self._states) <= 1: return 0
    num_prev_states = len(self._states) - 1
    penalty = 0
    for i in range(num_prev_states):
      penalty += monotonic_decreasing(self._states[i], self._states[-1], self._actions[i], self._actions[-1])

    return penalty / num_prev_states

  def step(self, action):
    ac = action[0] * ACTION_SPACE_NORMALIZER

    old_pos = self._positions[self._step_counts]
    old_price = self._prices[self._step_counts]
    old_price2 = self._prices[self._step_counts - 1]
    self._step_counts += 1
    new_pos = self._positions[self._step_counts] =\
      max(min(old_pos + ac, MAX_HOLDING), -MAX_HOLDING)
    new_price = self._prices[self._step_counts] = self._next_price(old_price, old_price2)

    trade = new_pos - old_pos
    cost = self._costs[self._step_counts] = TickSize * (abs(trade) + .01 * trade**2)
    PnL = self._pnls[self._step_counts] = (new_price - old_price) * old_pos - cost
    reward = self._rewards[self._step_counts] = PnL - .5 * kappa * PnL**2

    fn_penalty = 0
    if FUNC_PROPERTY_PENALTY: # incorporate function property
      self._states.append(new_price)
      self._actions.append(ac)
      fn_penalty = 5 / kappa * self._learn_func_property()

    info = {'pnl': PnL, 'cost': cost, 'reward': reward, 'penalty': fn_penalty}
    return self._get_state(), reward - fn_penalty, self._step_counts >= self._L + 1, info
