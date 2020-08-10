# RLF

All the custom environments lie in the folder `Environments`.

In the folder `Scripts`:
- The file `run_mean_reversion.py` has some simple/classical agents implemented and is mainly used for testing the custom environment is working;
- The file `run_mean_reversion_with_baselines.py` implements two OpenAI Baselines agents, PPO2 and A2C, that are chosen because of their ideal properties given our problem: they support continuous action spcace, recurrent policy, and multi processing.

### Instructions to train and run the Baselines agents
#### Example command
```python3 run_mean_reversion_with_baselines.py --agent=A2C --optimize=true```\
The `--optimize` flag will search for the optimal hyperparameters. This is usually the most time consuming part, so once you've found the optimal hyperparameters, you should pass them into the code and drop the `--optimize` flag.
If you have a saved model, you can comment out the training part and run the agent directly.

#### Advice
Due to limited computing power, the code is only using 100 trials to search for suitable hyperparameters, but this is already taking very long on a single machine. For better/optimal performance, I expect at least 1000 trials since DRL is mostly about tuning hyperparameters and therefore we should switch to cloud compute instances.
