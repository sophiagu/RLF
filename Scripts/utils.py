import gym

from stable_baselines.common.env_checker import check_env

def make_env(env_id):
  def _init():
    env = gym.make(env_id)
    check_env(env)
    return env
  return _init

def ppo2_params(trial):
  # See https://github.com/optuna/optuna/blob/master/optuna/trial/_trial.py for documentation.
  return {
    'ent_coef': trial.suggest_loguniform('ent_coef', .01, .3),
    'vf_coef': trial.suggest_uniform('vf_coef', .1, .9),
    'lam': trial.suggest_uniform('lam', .8, 1.),
  }
