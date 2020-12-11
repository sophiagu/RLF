# Reinforcment Learning for Finance (RLF)

This repo uses *off-the-shelf* technologies (`gym` for constructing envs and `stable-baselines` for training agents) and provides a general framework for training DRL agents for both `alpha-finding` and `hedging` strategies. To help you get started, I've included a few sample environments.

### Folder descriptions
All the sample environments are living under the folder `gym-rlf/`. Right now we have the following environments:
- `MeanReversionEnv`: A custom env for a security whose price process exhibits mean reversion property and its log value follows an Ornstein-Uhlenbeck process;
- `ARMAEnv`: Similar to `MeanReversionEnv` but uses an ARMA(2,1) process;
- `APTEnv`: A custom env for a single factor Arbitrage Pricing Theorem model.
- `DeltaHedgingEnv`: A custome env for delta hedging a European Call option with fixed strike and expiration.

The DRL agent lives in the folder `Scripts/`:
- `run_baselines_agent.py`: It implements the `PPO2` algorithm from `stable-baselines`. I chose `PPO2` because it combines ideas from both `A2C` and `TRPO`. More importantly, it supports continuous action/observation space, recurrent policy, and multi processing, which are desired properties for the common problems in quantitative finance.

### Prerequisites
```
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip
python3 -m pip install numpy scipy tensorflow gym stable-baselines optuna
sudo apt-get install sqlite3
```
Follow instructions in [stable-baselines-tf2](https://github.com/sophiagu/stable-baselines-tf2) to modify `stable-baselines`.\
`git clone` this repo.

**NOTE:**
- Before running the script, `cd` into `gym-rlf/` and register the envs:
```
python3 -m pip install -e .
```

### Instructions to train and run the Baselines agents

#### Example usage
```
python3 run_baselines_agent.py --env=mean_reversion --optimize=true
```
- For delta hedging problem, you can use a problem-specific Actor-Critic policy called `SigmoidMlpPolicy` by passing the flag `--use_sigmoid_layer=true`.
- If you want to penalize actions that do not obey a certain function property (not enforce), you can set `FUNC_PROPERTY_PENALTY` to `True` in the corresponding env.

### Troubleshoot
- `Check failed: PyBfloat16_Type.tp_base != nullptr`\
Follow https://www.programmersought.com/article/6151650908/ to uninstall and reinstall `numpy`.
