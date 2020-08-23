# RLF

All the custom environments are living under the folder `gym-rlf`. Right now we have the following environments:
- `MeanReversionEnv`: A custom env for a security whose price process exhibits mean reversion property and its log value follows an Ornstein-Uhlenbeck process;
- `APTEnv`: A custom env for a single factor Arbitrage Pricing Theorem model.

The folder `Scripts` contains:
- `run_baselines_agent.py`: It implements two OpenAI Baselines agents: `PPO2` and `A2C`. I chose them because they both support continuous action/observation space, recurrent policy, and multi processing, which are desired properties for our set of problems. The default algorithm is set to `PPO2` because it combines ideas from both `A2C` and `TRPO` (see [here](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html#id1)).

### Prerequisites
- Python 3
- Tensorflow 2.0
- Gym
- [stable-baselines](https://github.com/hill-a/stable-baselines) and replace part of it with [stable-baselines-tf2](https://github.com/sophiagu/stable-baselines-tf2)
- Optuna
- SQLite
```
python3 -m pip install numpy scipy tensorflow gym stable-baselines optuna
sudo apt-get install sqlite3
```

**NOTE:**
- Before running the script, `cd` into `gym-rlf/` and register the envs:
```
pip install -e .
```
- And `cd` into `Scripts/` to make a folder (`folder name` = `env_id` + '_plots') for saving the plots (*e.g.* for mean_reversion):
```
mkdir mean_reversion_plots
```

### Instructions to train and run the Baselines agents

#### Example usage
```
python3 run_baselines_agent.py --env=mean_reversion --optimize=true
```

#### Tips
- The `--optimize` flag will search for the optimal hyperparameters. This is usually the most time consuming part, so once you've found the optimal hyperparameters, you should set the `--optimize` flag to `false` and let the code load the best hyperparameters from the db;
- If you have a saved model, you can comment out the training part and run the agent directly;
- Due to limited computing power, the code is only using 100 trials to search for suitable hyperparameters, but it's already taking very long on a single machine. For better/optimal performance, I expect at least 1000 trials since as of today DRL is still [very unstable](https://www.alexirpan.com/2018/02/14/rl-hard.html) and carefully tuning the hyperparameters seems to be one of very few ways to reduce the pain. So if possible we should switch to use cloud compute instances.
