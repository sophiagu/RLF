# RLF

All the custom environments are living under the folder `gym-rlf`. Right now we have the following environments:
- `MeanReversionEnv`: A custom env for a security whose price process exhibits mean reversion property and its log value follows an Ornstein-Uhlenbeck process;
- `APTEnv`: A custom env for a single factor Arbitrage Pricing Theorem model.

The folder `Scripts` contains:
- `run_baselines_agent.py`: It implements two OpenAI Baselines agents: `PPO2` and `A2C`. I chose them because they both support continuous action/observation space, recurrent policy, and multi processing, which are desired properties for our set of problems. The default algorithm is set to `PPO2` because it combines ideas from both `A2C` and `TRPO` (see [here](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html#id1)).

### Prerequisites
```
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip
python3 -m pip install numpy scipy tensorflow gym stable-baselines optuna
sudo apt-get install sqlite3
```
Follow instructions in [stable-baselines-tf2](https://github.com/sophiagu/stable-baselines-tf2) to modify `stable-baselines`.

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
- The `--optimize` flag will search for the optimal hyperparameters. This is usually the most time consuming part, so once you've found the optimal hyperparameters, you should drop the `--optimize` flag and let the code load the best hyperparameters from the db;
- The hyperparameters need to be `--optimize`d at least once;
- If you have a saved model, you can comment out the training part and run the agent directly.
