import gym
import os
from stable_baselines3 import PPO
import highway_env

env = gym.make("parking-v0")

env.reset()

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

model = PPO(policy="MultiInputPolicy", verbose=1, env=env, tensorboard_log=logdir)

TIMESTAMPS = 10000

for i in range(1,50):
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTAMPS*i}")