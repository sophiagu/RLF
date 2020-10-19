import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np

from gym import spaces
from gym_rlf.envs.fn_properties import monotonic_decreasing
from gym_rlf.envs.rlf_env import RLFEnv, action_space_normalizer, MIN_PRICE, MAX_PRICE
from gym_rlf.envs.Parameters import TickSize, OptionSize, S0, sigma_dh, kappa_dh
from scipy.stats import norm

FUNC_PROPERTY_PENALTY = True
IS_EPISODIC = True # this must be True if FUNC_PROPERTY_PENALTY is True

    
def BSM_call_price_and_delta(K, tau, St, sigma):
  if tau <= 0: return 0, 0
  # assuming zero interest rate
  numerator = math.log(St / K) + (.5 * sigma**2) * tau
  denominator = sigma * math.sqrt(tau)
  d1 = numerator / denominator
  d2 = d1 - denominator
  price = St * norm.cdf(d1) - K * norm.cdf(d2)
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
      high=np.array([OptionSize, self._L, MAX_PRICE, MAX_PRICE]),
      shape=(4,))

  def _next_price(self):
    rn = np.random.normal(0, 1., 1)[0]
    x = -.5 * sigma_dh**2 * self._step_counts + sigma_dh * rn
    p = S0 * np.exp(x)
    p = min(p, MAX_PRICE)
    p = max(p, MIN_PRICE)
    return p

  def _get_state(self):
    return np.array([
      self._positions[self._step_counts % self._L],
      (self._L - self._step_counts % self._L) % self._L,
      self._prices[(self._step_counts - 1) % self._L],
      self._prices[self._step_counts % self._L],
    ])

  def _learn_func_property(self):
    if len(self._states) <= 1: return 0
    num_prev_states = len(self._states) - 1
    penalty = 0
    for i in range(num_prev_states):
      penalty += monotonic_decreasing(self._states[i], self._states[-1], self._actions[i], self._actions[-1])

    return penalty / num_prev_states

  def reset(self):
    super(DeltaHedgingEnv, self).reset()

    self._prices[0] = self._prices[1] = S0
    self._option_prices = np.zeros(self._L)
    self._option_prices[0] = self._option_prices[1] = BSM_call_price_and_delta(S0, self._L - 1, S0, sigma_dh)[0]
    return self._get_state()

  def step(self, action):
    ac = round(action[0] * action_space_normalizer)

    old_pos = self._positions[self._step_counts % self._L]
    old_price = self._prices[self._step_counts % self._L]
    old_option_price = self._option_prices[self._step_counts % self._L]
    self._step_counts += 1
    new_pos = self._positions[self._step_counts % self._L] = max(min(old_pos + ac, OptionSize), -OptionSize)
    new_price = self._prices[self._step_counts % self._L] = self._next_price()
    new_option_price = self._option_prices[self._step_counts % self._L] =\
      BSM_call_price_and_delta(S0,  (self._L - self._step_counts % self._L) % self._L, new_price, sigma_dh)[0]

    trade_size = abs(new_pos - old_pos)
    cost = self._costs[self._step_counts % self._L] = TickSize * (trade_size + .01 * trade_size**2)
    PnL = self._pnls[self._step_counts % self._L] = (new_price - old_price) * old_pos + (new_option_price - old_option_price) - cost
    reward = PnL - .5 * kappa_dh * PnL**2

    fn_penalty = 0
    if FUNC_PROPERTY_PENALTY: # incorporate function property
      self._states.append(new_price)
      self._actions.append(ac)
      fn_penalty = abs(reward) * self._learn_func_property()
      
    done = self._step_counts == self._L if IS_EPISODIC else False
    info = {'pnl': PnL, 'cost': cost, 'reward': reward, 'penalty': fn_penalty}
    return self._get_state(), reward - fn_penalty, done, info
 
  def render(self, mode='human'):
    super(DeltaHedgingEnv, self).render()

    t = np.linspace(0, self._L, self._L)
    fig, axs = plt.subplots(3, 1, figsize=(16, 24), constrained_layout=True)
    axs[0].plot(t, self._prices, label='stock prices')
    axs[0].plot(t, self._option_prices, label='option prices')
    axs[1].plot(t, self._positions)
    axs[2].plot(t, np.cumsum(self._pnls))
    axs[0].set_ylabel('price')
    axs[1].set_ylabel('position')
    axs[2].set_ylabel('cumulative P/L')
    axs[0].legend()
    plt.title('Out-of-sample simulation of RL agent')
    plt.xlabel('steps')
    plt.savefig('{}/plot_{}.png'.format(self._folder_name, self._render_counts))
    plt.close()
    plt.plot(t, np.cumsum(self._costs), label='cumulative costs')
    plt.plot(t, np.cumsum(self._pnls + self._costs), label='cumulative profits')
    plt.legend()
    plt.savefig('{}/costs_and_profits_plot_{}.png'.format(self._folder_name, self._render_counts))
    plt.close()
