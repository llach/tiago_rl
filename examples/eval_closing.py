import os

from tiago_rl.envs import GripperTactileEnv
from tiago_rl.misc.load_cell_vis import LoadCellVisualiser

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

with_vis = True

# Environment setup
# ----------------------------
env = GripperTactileEnv(**{"render_mode": "human"} if with_vis else {})
env = TimeLimit(env, max_episode_steps=300)
vis = LoadCellVisualiser(env) if with_vis else None

# Load model
# ----------------------------
config_name = "/tmp/"
model = PPO('MlpPolicy', env, verbose=1)
model = model.load("/tmp/best_model.zip")

for trial in range(10):
    obs, _ = env.reset()
    if vis: vis.reset()
    done = False
    crew = 0

    while not done:
        pred = model.predict(obs, deterministic=True)
        action = pred[0]
        # action = [0.0,0.0]
        # print(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        crew += reward

        if vis: vis.update_plot(action=action, reward=reward)
    print(crew)
