import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gym import spaces
from gym_rlf.envs.rlf_env import RLFEnv, action_space_normalizer, MAX_HOLDING, MIN_PRICE, MAX_PRICE
from gym_rlf.envs.Parameters import TickSize, sigma, kappa, alpha, factor_alpha, factor_sensitivity, factor_sigma, p_e

IS_EPISODIC = True


class APTEnv(RLFEnv):
  def __init__(self):
    super(APTEnv, self).__init__('apt_plots/')

    # Use a Box to represent the action space with the first param being
    # (trade of the security) and the second param being (trade of the factor security).
    self.action_space = spaces.Box(
      low=np.array([-1, -1]),
      high=np.array([1, 1]),
      shape=(2,))
    # Use a Box to represent the observation space with params: (position of the security),
    # (position of the factor security) and (price of the security).
    # The price of the factor security is hidden.
    self.observation_space = spaces.Box(
      low=np.array([-MAX_HOLDING, -MAX_HOLDING, MIN_PRICE]),
      high=np.array([MAX_HOLDING, MAX_HOLDING, MAX_PRICE]),
      shape=(3,))

  def _next_price(self, p, p_f):
    rn1 = np.random.normal(0, 1., 1)[0]
    rn2 = np.random.normal(0, 1., 1)[0]

    factor_return = factor_alpha + factor_sigma * rn1
    p_f_new = (1 + factor_return) * p_f
    p_f_new = min(p_f_new, MAX_PRICE)
    p_f_new = max(p_f_new, MIN_PRICE)

    r = alpha + factor_sensitivity * factor_return + sigma * rn2
    p_new = (1 + r) * p
    p_new = min(p_new, MAX_PRICE)
    p_new = max(p_new, MIN_PRICE)
    return p_new, p_f_new

  def reset(self):
    super(APTEnv, self).reset()

    self._factor_prices = np.zeros(self._L+1)
    self._factor_prices[0] = p_e
    self._factor_positions = np.zeros(self._L+1)
    return self._get_state()

  def _get_state(self):
    return np.array([self._positions[self._step_counts % self._L],
                     self._factor_positions[self._step_counts % self._L],
                     self._prices[self._step_counts % self._L]])
    
  def step(self, action):
    ac1 = round(action[0] * action_space_normalizer)
    ac2 = round(action[1] * action_space_normalizer)

    old_pos = self._positions[self._step_counts % self._L]
    old_factor_pos = self._factor_positions[self._step_counts % self._L]
    old_price = self._prices[self._step_counts % self._L]
    old_factor_price = self._factor_prices[self._step_counts % self._L]
    self._step_counts += 1
    new_pos = self._positions[self._step_counts % self._L] =\
      max(min(old_pos + ac1, MAX_HOLDING), -MAX_HOLDING)
    new_factor_pos = self._factor_positions[self._step_counts % self._L] =\
      max(min(old_factor_pos + ac2, MAX_HOLDING), -MAX_HOLDING)
    new_price, new_factor_price =\
      self._prices[self._step_counts % self._L], self._factor_prices[self._step_counts % self._L] =\
      self._next_price(old_price, old_factor_price)

    trade_size = abs(new_pos - old_pos) + abs(new_factor_pos - old_factor_pos)
    cost = TickSize * (trade_size + 1e-2 * trade_size**2)
    PnL = (new_price - old_price) * old_pos + (new_factor_price - old_factor_price) * old_factor_pos - cost
    self._costs[self._step_counts % self._L] = cost
    self._profits[self._step_counts % self._L] = PnL + cost
    self._rewards[self._step_counts % self._L] = PnL - .5 * kappa * PnL**2

    done = self._step_counts == self._L if IS_EPISODIC else False
    return self._get_state(), self._rewards[self._step_counts % self._L], done, {}
 
  def render(self, mode='human'):
    super(APTEnv, self).render()

    t = np.linspace(0, self._L, self._L+1)
    fig, axs = plt.subplots(5, 1, figsize=(16, 40), constrained_layout=True)
    axs[0].plot(t, self._prices)
    axs[1].plot(t, self._factor_prices)
    axs[2].plot(t, self._positions)
    axs[3].plot(t, self._factor_positions)
    axs[4].plot(t, np.cumsum(self._rewards))
    axs[0].set_ylabel('price')
    axs[1].set_ylabel('factor price')
    axs[2].set_ylabel('position')
    axs[3].set_ylabel('factor position')
    axs[4].set_ylabel('cumulative P/L')
    plt.title('Out-of-sample simulation of RL agent')
    plt.xlabel('steps')
    plt.savefig('{}/plot_{}.png'.format(self._folder_name, self._render_counts))
    plt.close()
    plt.plot(t, np.cumsum(self._costs), label='cumulative costs')
    plt.plot(t, np.cumsum(self._profits), label='cumulative profits')
    plt.legend()
    plt.savefig('{}/costs_and_profits_plot_{}.png'.format(self._folder_name, self._render_counts))
    plt.close()
