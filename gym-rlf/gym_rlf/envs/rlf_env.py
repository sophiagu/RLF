import gym
from gym import error, spaces, utils
from gym.utils import seeding

class RLFEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    pass
  def step(self, action):
    raise NotImplementedError
  def reset(self):
    raise NotImplementedError
  def render(self, mode='human'):
    pass
  def close(self):
    pass