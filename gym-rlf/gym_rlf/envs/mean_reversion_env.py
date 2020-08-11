import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gym import spaces
from gym_rlf.envs.rlf_env import RLFEnv, action_space_normalizer, MAX_HOLDING, MIN_PRICE, MAX_PRICE
from gym_rlf.envs.Parameters import TickSize, Lambda, sigma, kappa, p_e


class MeanReversionEnv(RLFEnv):
  def __init__(self):
    super(MeanReversionEnv, self).__init__(1000, 'mean_reversion_plots/')

    self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
    # Use a Box to represent an observation space with the first param being (position)
    # and the second param being (price).
    self.observation_space = spaces.Box(
      low=np.array([-MAX_HOLDING, MIN_PRICE]),
      high=np.array([MAX_HOLDING, MAX_PRICE]),
      shape=(2,))

  def next_price(self, p):
    x = np.log(p / p_e)
    rn = np.random.normal(0, 1.0, 1)[0]
    x = x - Lambda * x + sigma * rn
    p_new = p_e * np.exp(x)
    p_new = min(p_new, MAX_PRICE)
    p_new = max(p_new, MIN_PRICE)
    return p_new

  def reset(self):
    super(MeanReversionEnv, self).reset()

    return self.get_state()

  def get_state(self):
    return np.array([self.positions[self.step_counts], self.prices[self.step_counts]])
    
  def step(self, action):
    ac = round(action[0] * action_space_normalizer)

    old_pos = self.positions[self.step_counts]
    old_price = self.prices[self.step_counts]
    self.step_counts += 1
    new_pos = self.positions[self.step_counts] = max(min(old_pos + ac, MAX_HOLDING), -MAX_HOLDING)
    new_price = self.prices[self.step_counts] = self.next_price(old_price)

    trade_size = abs(new_pos - old_pos)
    cost = TickSize * (trade_size + 1e-2 * trade_size**2)
    PnL = (new_price - old_price) * old_pos - cost
    self.costs[self.step_counts] = cost
    self.profits[self.step_counts] = PnL + cost
    self.rewards[self.step_counts] = PnL - .5 * kappa * PnL**2

    done = self.step_counts == self.L
    if done: # liquidate the remaining new_pos shares of the security
      add_cost = TickSize * (abs(new_pos) + 1e-2 * new_pos**2)
      add_PnL = new_price * new_pos - add_cost
      self.costs[self.step_counts] += add_cost
      self.profits[self.step_counts] += add_PnL + add_cost
      self.rewards[self.step_counts] += add_PnL - .5 * kappa * add_PnL**2

    return self.get_state(), self.rewards[self.step_counts], done, {}
 
  def render(self, mode='human'):
    super(MeanReversionEnv, self).render()

    t = np.linspace(0, self.L, self.L+1)
    fig, axs = plt.subplots(3, 1, figsize=(16, 24), constrained_layout=True)
    axs[0].plot(t, self.prices)
    axs[1].plot(t, self.positions)
    axs[2].plot(t, np.cumsum(self.rewards))
    axs[0].set_ylabel('price')
    axs[1].set_ylabel('position')
    axs[2].set_ylabel('cumulative P/L')
    plt.title('Out-of-sample simulation of RL agent')
    plt.xlabel('steps')
    plt.savefig('{}/plot_{}.png'.format(self.folder_name, self.render_counts))
    plt.close()
    plt.plot(t, np.cumsum(self.costs), label='cumulative costs')
    plt.plot(t, np.cumsum(self.profits), label='cumulative profits')
    plt.legend()
    plt.savefig('{}/costs_and_profits_plot_{}.png'.format(self.folder_name, self.render_counts))
    plt.close()
