import gym
from stable_baselines3 import PPO, A2C
import highway_env

env = gym.make("parking-v0")
env.reset()

models_dir = "models/A2C"
model_path = f"{models_dir}/180000.zip"
model = A2C.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    observation = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)