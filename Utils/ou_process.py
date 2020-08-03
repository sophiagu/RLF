import numpy as np


def get_ou_sequence(theta, sigma, length, low, high):
  '''
  Generates an Ornstein-Uhlenbeck process:
  dx_t = (-theta * x_t)dt  + sigma dW_t

  Modified from https://www.pik-potsdam.de/members/franke/lecture-sose-2016/introduction-to-python.pdf.
  '''
  t = np.linspace(low, high, length)
  dt = np.mean(np.diff(t))
  
  x = np.zeros(length)
  x[0] = 0. # start from the equilibrium price
  
  drift = lambda x, t: -theta * x
  diffusion = lambda x, t: sigma
  noise = np.random.normal(loc=0.0, scale=1.0, size=length) * np.sqrt(dt)

  for i in range(1,length):
    x[i] = x[i-1] + drift(x[i-1], i*dt) * dt + diffusion(x[i-1], i*dt) * noise[i]
    
  return x
