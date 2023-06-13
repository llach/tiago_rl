import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, TD3, DDPG
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

from tiago_rl.envs import GripperTactileEnv
from tiago_experiments import SaveOnBestTrainingRewardCallback

from gymnasium.wrappers import TimeLimit

# Create log dir
timesteps = 2e5
log_dir = "/tmp/tactile/"
os.makedirs(log_dir, exist_ok=True)

# Environment setup
# ----------------------------
env = GripperTactileEnv(alpha=2, delta=2.0, fgoal_range=[0.3, 0.6])
env = TimeLimit(env, max_episode_steps=250)
env = Monitor(env, log_dir)

model = TD3('MlpPolicy', env, verbose=1)
callback = SaveOnBestTrainingRewardCallback(
    env=env,
    check_freq=2000,
    total_steps=timesteps,
    save_path=log_dir,
    step_offset=1e3,
    mean_n=100
)

# Train the agent
model.learn(total_timesteps=int(timesteps), callback=callback)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Gripper Closing Environment")
plt.show()