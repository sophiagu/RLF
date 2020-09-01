import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gym import spaces
from gym_rlf.envs.rlf_env import RLFEnv, action_space_normalizer, MAX_HOLDING, MIN_PRICE, MAX_PRICE
from gym_rlf.envs.Parameters import TickSize, Lambda, sigma, kappa, p_e

L2_REGULARIZED_REWARD = True
IS_EPISODIC = True # this must be True if using l2 regularized reward
PENALTY_WEIGHT = 1000
MAX_PENALTY = 5000

def func_property(s1, s2, a1, a2):
  if (a1 - a2) * (s1 - s2) <= 0: return 0
  return min(MAX_PENALTY, (a1 - a2)**2)
    

class MeanReversionEnv(RLFEnv):
  def __init__(self):
    super(MeanReversionEnv, self).__init__('mean_reversion_plots/')

    self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
    # Use a Box to represent the observation space with the first param being (position)
    # and the second param being (price).
    self.observation_space = spaces.Box(
      low=np.array([-MAX_HOLDING, MIN_PRICE]),
      high=np.array([MAX_HOLDING, MAX_PRICE]),
      shape=(2,))

  def _next_price(self, p):
    x = np.log(p / p_e)
    rn = np.random.normal(0, 1., 1)[0]
    x = x - Lambda * x + sigma * rn
    p_new = p_e * np.exp(x)
    p_new = min(p_new, MAX_PRICE)
    p_new = max(p_new, MIN_PRICE)
    return p_new

  def reset(self):
    super(MeanReversionEnv, self).reset()

    return self._get_state()

  def _get_state(self):
    return np.array([
      self._positions[self._step_counts % self._L],
      self._prices[self._step_counts % self._L]])
    
  def _learn_func_property(self, func):
    if len(self._states) <= 1: return 0
    penalty = 0
    for i in range(len(self._states) - 1):
      penalty += func(self._states[i], self._states[-1], self._actions[i], self._actions[-1])
    return penalty

  def step(self, action):
    ac = round(action[0] * action_space_normalizer)

    old_pos = self._positions[self._step_counts % self._L]
    old_price = self._prices[self._step_counts % self._L]
    self._step_counts += 1
    new_pos = self._positions[self._step_counts % self._L] =\
      max(min(old_pos + ac, MAX_HOLDING), -MAX_HOLDING)
    new_price = self._prices[self._step_counts % self._L] = self._next_price(old_price)

    trade_size = abs(new_pos - old_pos)
    cost = TickSize * (trade_size + 1e-2 * trade_size**2)
    PnL = (new_price - old_price) * old_pos - cost
    self._costs[self._step_counts % self._L] = cost
    self._profits[self._step_counts % self._L] = PnL + cost
    self._rewards[self._step_counts % self._L] = PnL - .5 * kappa * PnL**2

    done = self._step_counts == self._L if IS_EPISODIC else False
    reward = self._rewards[self._step_counts % self._L]
    if L2_REGULARIZED_REWARD: # incorporate function property
      self._states.append(new_price)
      if abs(old_pos) > 0:
        self._actions.append(ac/old_pos)
      else:
        self._actions.append(ac)
      reward -= PENALTY_WEIGHT * self._learn_func_property(func_property)
    return self._get_state(), reward, done, {}
 
  def render(self, mode='human'):
    super(MeanReversionEnv, self).render()

    t = np.linspace(0, self._L, self._L+1)
    fig, axs = plt.subplots(3, 1, figsize=(16, 24), constrained_layout=True)
    axs[0].plot(t, self._prices)
    axs[1].plot(t, self._positions)
    axs[2].plot(t, np.cumsum(self._rewards))
    axs[0].set_ylabel('price')
    axs[1].set_ylabel('position')
    axs[2].set_ylabel('cumulative P/L')
    plt.title('Out-of-sample simulation of RL agent')
    plt.xlabel('steps')
    plt.savefig('{}/plot_{}.png'.format(self._folder_name, self._render_counts))
    plt.close()
    plt.plot(t, np.cumsum(self._costs), label='cumulative costs')
    plt.plot(t, np.cumsum(self._profits), label='cumulative profits')
    plt.legend()
    plt.savefig('{}/costs_and_profits_plot_{}.png'.format(self._folder_name, self._render_counts))
    plt.close()
