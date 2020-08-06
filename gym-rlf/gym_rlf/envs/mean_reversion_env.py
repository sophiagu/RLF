import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

from gym import spaces
from gym_rlf.envs.rlf_env import RLFEnv
from Utils.ou_process import get_ou_sequence


class MeanReversionEnv(RLFEnv):
  ACTION = {
      0 : -100,
      1 : +100,
  }

  def __init__(self):
    super(MeanReversionEnv, self).__init__()
    self.temp_name = next(tempfile._get_candidate_names())
    
    self.equilibrium_price = 100.
    self.theta, self.sigma, self.kappa, self.length, self.low, self.high = .1, .2, 1.0e-4, 1000, -2., .301
    self.linear_cost_multiplier, self.quadratic_cost_multiplier = .2, .15
    self.render_count = 0
    
    # self.observation_space = spaces.Dict({
    #   'position': spaces.Box(low=-100*self.length, high=100*self.length, shape=(1,)),
    #   'price': spaces.Box(low=0., high=self.equilibrium_price * 10.**self.high, shape=(1,)),
    # })
    # Use a Box to represent an observation splace with the first param being position and the second being price.
    self.observation_space = spaces.Box(
      low=np.array([-100*self.length, 0,]),
      high=np.array([100*self.length, self.equilibrium_price * 10.**self.high]),
      shape=(2,),
      dtype=np.float32)
    
    # simple discrete action space
    # self.action_space = spaces.Discrete(2) # buy or sell 100 shares of the underlying
    
    # continuous action space
    self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    # OpenAI baselines/stable-baselines recommends to normalize continuous action space
    self.action_space_normalizer = 100
  
  def reset(self):
    self.count = 0
    self.price_process = get_ou_sequence(self.theta, self.sigma, self.length+1, self.low, self.high)
    self.price = self.equilibrium_price
    self.position = np.zeros(self.length+1)
    self.position[0] = 1000
    self.cumulative_rewards = [0]
    
    return self.getState()

  def getState(self):
    # return {'position': self.position[self.count], 'price': self.price}
    return np.array([self.position[self.count], self.price]).astype(np.float32)

  def step(self, action):
    # make sure we only trade integer number of shares
    a = round(action[0] * self.action_space_normalizer)
    # bound the action
    a = min(100*self.length-self.position[self.count], a) if\
     self.position[self.count] > 0 else min(100*self.length, a)
    # a = max(-100*self.length-self.position[self.count], a)
    # if long only: make sure we don't own negative number of shares
    a = max(-self.position[self.count], a)
    new_price = self.equilibrium_price * 10**self.price_process[self.count]
    profit = (new_price - self.price) * self.position[self.count]
    cost = a * new_price
    impact = self.linear_cost_multiplier * np.abs(a) + self.quadratic_cost_multiplier * a**2
    PnL = profit - cost - impact
    reward = PnL - self.kappa * PnL**2
    
    self.count += 1
    done = self.count % self.length == 0

    self.price = new_price
    # self.position[self.count] = self.position[self.count-1] + MeanReversionEnv.ACTION[action[0]]
    self.position[self.count] = self.position[self.count-1] + a
    if done:
      PnL = self.price * self.position[self.count] - self.linear_cost_multiplier * np.abs(self.position[self.count])\
       + self.quadratic_cost_multiplier * self.position[self.count]**2
      reward += PnL - self.kappa * PnL**2
    self.cumulative_rewards.append(self.cumulative_rewards[-1] + reward)

    return self.getState(), reward, done, {}
 
  def render(self, mode='human'):
    if mode == 'human':
      folder_name = 'mean_reversion_plots/{}'.format(self.temp_name)
      if not os.path.exists(folder_name):
        os.makedirs(folder_name)
            
      if (self.render_count+1) % 1 == 0: # only render every x step(s)
        t = np.linspace(0, self.length, self.length+1)
        fig, axs = plt.subplots(3, 1, figsize=(16, 24), constrained_layout=True)

        axs[0].plot(t, self.equilibrium_price * 10**self.price_process, label='price')
        axs[1].plot(t, self.position, label='position')
        axs[2].plot(t, self.cumulative_rewards, label='cumulative P/L')
        axs[0].set_ylabel('price')
        axs[1].set_ylabel('position')
        axs[2].set_ylabel('cumulative P/L')
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        plt.title('Out-of-sample simulation of RL agent')
        plt.xlabel('steps')
        plt.savefig('mean_reversion_plots/{}/plot_{}.png'.format(self.temp_name, self.render_count))
        plt.close()

      self.render_count += 1
