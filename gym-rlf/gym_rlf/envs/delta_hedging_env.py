import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np

from gym import spaces
from gym_rlf.envs.fn_properties import convex, monotonic_decreasing
from gym_rlf.envs.rlf_env import RLFEnv, MIN_PRICE, MAX_PRICE
from gym_rlf.envs.Parameters import TickSize, OptionSize, S0, sigma_dh, kappa_dh
from scipy.stats import norm

# Stable Baselines recommends to normalize continuous action space because the Baselines
# agents only sample actions from a standard Gaussian.
# We use a space normalizer to rescale the action space to [-OptionSize, OptionSize].
ACTION_SPACE_NORMALIZER = OptionSize
FUNC_PROPERTY_PENALTY = False

def BSM_call_price_and_delta(K, tau, St, sigma):
  numerator = math.log(St / K) + (.5 * sigma**2) * tau
  if tau > 0:
    denominator = sigma * math.sqrt(tau)
    d1 = numerator / denominator
  else:
    denominator = 0.
    d1 = float('inf')
  d2 = d1 - denominator
  price = max(St * norm.cdf(d1) - K * norm.cdf(d2), 0.)
  delta = norm.cdf(d1)
  return price, delta
    

class DeltaHedgingEnv(RLFEnv):
  def __init__(self):
    super(DeltaHedgingEnv, self).__init__('delta_hedging_plots/')

    self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
    # Use a Box to represent the observation space with params:
    # (underlying position), (time to maturity), (previous underlying price) and (current underlying price).
    self.observation_space = spaces.Box(
      low=np.array([-OptionSize, 0, MIN_PRICE, MIN_PRICE]),
      high=np.array([0, self._L, MAX_PRICE, MAX_PRICE]),
      shape=(4,))

  def _next_price(self, p):
    rn = np.random.normal(0, 1., 1)[0]
    p_new = p * np.exp(sigma_dh * rn)
    p_new = min(p_new, MAX_PRICE)
    p_new = max(p_new, MIN_PRICE)
    return p_new

  def _get_state(self):
    return np.array([
      self._positions[self._step_counts],
      self._L + 1 - self._step_counts,
      self._prices[self._step_counts - 1],
      self._prices[self._step_counts],
    ])

  def _learn_func_property(self):
    if len(self._states) <= 1: return 0
    num_prev_states = len(self._states) - 1
    penalty = 0
    for i in range(num_prev_states):
      penalty += monotonic_decreasing(self._states[i], self._states[-1], self._actions[i], self._actions[-1])

    return penalty / num_prev_states

  # def _learn_func_property(self):
  #   if len(self._states) <= 2: return 0
  #   num_prev_states = len(self._states) - 2
  #   penalty = 0
  #   for i in range(num_prev_states):
  #     for j in range(i + 1, num_prev_states + 1):
  #       min_id, max_id = i, j
  #       if self._states[j] < self._states[min_id]: min_id = j
  #       if self._states[-1] < self._states[min_id]: min_id = -1
  #       if self._states[i] > self._states[max_id]: max_id = i
  #       if self._states[-1] > self._states[max_id]: max_id = -1
  #       mid_id = i + j - 1 - min_id - max_id
  #       assert mid_id in [i, j, -1], 'Invalid mid_id={}'.format(mid_id)
  #       penalty += convex(self._states[min_id], self._states[mid_id], self._states[max_id],
  #                         self._actions[min_id], self._actions[mid_id], self._actions[max_id])
  #
  #   return penalty / (num_prev_states * (num_prev_states + 1))

  def reset(self):
    super(DeltaHedgingEnv, self).reset()

    self._prices[0] = self._prices[1] = S0
    self._option_prices = np.zeros(self._L + 2)
    self._benchmark_positions = np.zeros(self._L + 2)
    self._bm_pnls = np.zeros(self._L + 2)

    option_price, delta = BSM_call_price_and_delta(S0, self._L, S0, sigma_dh)
    self._option_prices[0] = self._option_prices[1] = option_price
    self._positions[0] = self._positions[1] = -delta * OptionSize
    self._benchmark_positions[0] = self._benchmark_positions[1] = -delta * OptionSize
    return self._get_state()

  def step(self, action):
    ac = action[0] * ACTION_SPACE_NORMALIZER
    
    old_pos = self._positions[self._step_counts]
    old_bm_pos = self._benchmark_positions[self._step_counts]
    old_price = self._prices[self._step_counts]
    old_option_price = self._option_prices[self._step_counts]

    self._step_counts += 1
    new_pos = self._positions[self._step_counts] = max(min(old_pos + ac, 0), -OptionSize)
    new_price = self._prices[self._step_counts] = self._next_price(old_price)
    new_option_price, delta =\
      BSM_call_price_and_delta(S0,  self._L + 1 - self._step_counts, new_price, sigma_dh)
    self._option_prices[self._step_counts] = new_option_price
    new_bm_pos = self._benchmark_positions[self._step_counts] = -OptionSize * delta

    trade, bm_trade = new_pos - old_pos, new_bm_pos - old_bm_pos
    cost = self._costs[self._step_counts] = TickSize * (abs(trade) + .01 * trade**2)
    PnL = self._pnls[self._step_counts] =\
      (new_price - old_price) * old_pos + (new_option_price - old_option_price) * OptionSize - cost
    reward = self._rewards[self._step_counts] = PnL - .5 * kappa_dh * PnL**2
    
    bm_cost = TickSize * (abs(bm_trade) + .01 * bm_trade**2)
    self._bm_pnls[self._step_counts] =\
      (new_price - old_price) * old_bm_pos + (new_option_price - old_option_price) * OptionSize - bm_cost

    fn_penalty = 0
    if FUNC_PROPERTY_PENALTY: # incorporate function property
      self._states.append(new_price)
      self._actions.append(ac)
      fn_penalty = 5 / kappa_dh * self._learn_func_property()

    info = {'pnl': PnL, 'cost': cost, 'reward': reward, 'penalty': fn_penalty}
    return self._get_state(), reward - fn_penalty, self._step_counts >= self._L + 1, info
 
  def render(self, mode='human'):
    super(DeltaHedgingEnv, self).render()

    t = np.linspace(0, self._L + 1, self._L + 2)
    fig, axs = plt.subplots(4, 1, figsize=(16, 32), constrained_layout=True)
    axs[0].plot(t, self._prices)
    axs[1].plot(t, self._option_prices)
    axs[2].plot(t, self._positions, label='position')
    axs[2].plot(t, self._benchmark_positions, label='benchmark position')
    axs[3].plot(t, np.cumsum(self._pnls), label='P/L')
    axs[3].plot(t, np.cumsum(self._bm_pnls), label='benchmark P/L')
    axs[0].set_ylabel('stock price')
    axs[1].set_ylabel('option price')
    axs[2].set_ylabel('position')
    axs[3].set_ylabel('cumulative P/L')
    axs[2].legend()
    axs[3].legend()
    plt.title('Out-of-sample simulation of RL agent')
    plt.xlabel('steps')
    plt.savefig('{}/plot_{}.png'.format(self._folder_name, self._render_counts))
    plt.close()

    _, axs2 = plt.subplots(4, 1, figsize=(16, 32), constrained_layout=True)
    axs2[0].plot(t, self._rewards)
    axs2[1].plot(t, np.cumsum(self._rewards))
    axs2[2].plot(t, np.cumsum(self._costs))
    axs2[3].plot(t, np.cumsum(self._pnls + self._costs))
    axs2[0].set_ylabel('reward per timestep')
    axs2[1].set_ylabel('cumulative reward')
    axs2[2].set_ylabel('cumulative costs')
    axs2[3].set_ylabel('cumulative revenues')
    plt.title('Out-of-sample reward and cost of RL agent')
    plt.xlabel('steps')
    plt.savefig('{}/reward_and_cost_{}.png'.format(self._folder_name, self._render_counts))
    plt.close()
