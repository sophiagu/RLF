import gym

from stable_baselines.common.env_checker import check_env

def make_env(env_id):
  def _init():
    env = gym.make(env_id)
    check_env(env)
    return env
  return _init

def ppo2_params(trial):
  return {
    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
    'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
    'vf_coef': trial.suggest_uniform('vf_coef', .1, .9),
    'lam': trial.suggest_uniform('lam', .8, 1.)
  }