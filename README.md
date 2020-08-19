# RLF

All the custom environments are living under the folder `gym-rlf`. Right now we have the following environments:
- `MeanReversionEnv`: A custom env for a security whose price process exhibits mean reversion property and its log value follows an Ornstein-Uhlenbeck process;
- `APTEnv`: A custom env for a single factor Arbitrage Pricing Theorem model.

The folder `Scripts` contains:
- `run_baselines_agent.py`: It implements two OpenAI Baselines agents: `PPO2` and `A2C`. I chose them because they both support continuous action/observation space, recurrent policy, and multi processing, which are desired properties for our set of problems.

### Prerequisites
- Python 3
- Tensorflow 2.0
- Gym
- [stable-baselines](https://github.com/hill-a/stable-baselines) and [stable-baselines-tf2](https://github.com/sophiagu/stable-baselines-tf2)
- Optuna
- SQLite

### Instructions to train and run the Baselines agents
#### Example usage
```python3 run_baselines_agent.py --env=mean_reversion --agent=A2C --optimize=true --num_trials=100 --num_eps=10 --evaluation_steps=500000 --train_steps=1000000```

#### Tips
- The `--optimize` flag will search for the optimal hyperparameters. This is usually the most time consuming part, so once you've found the optimal hyperparameters, you should set the `--optimize` flag to `false` and let the code load the best hyperparameters from the db;
- If you have a saved model, you can comment out the training part and run the agent directly;
- Due to limited computing power, the code is only using 100 trials to search for suitable hyperparameters, but it's already taking very long on a single machine. For better/optimal performance, I expect at least 1000 trials since as of today DRL is still [very unstable](https://www.alexirpan.com/2018/02/14/rl-hard.html) and carefully tuning the hyperparameters seems to be one of very few ways to reduce the pain. So if possible we should switch to use cloud compute instances.
