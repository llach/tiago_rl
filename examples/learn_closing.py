import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

from tiago_rl.envs import GripperTactileCloseEnv
from tiago_rl.misc import SaveOnBestTrainingRewardCallback

from gym.wrappers import TimeLimit

# Create log dir
log_dir = "/tmp/"
os.makedirs(log_dir, exist_ok=True)

# Environment setup
# ----------------------------
force_type = 'binary'
target_forces = np.array([1.0, 1.0])

env = GripperTactileCloseEnv(target_forces=target_forces, force_type=force_type)
env = TimeLimit(env, max_episode_steps=300)
env = Monitor(env, log_dir)

model = PPO('MlpPolicy', env, verbose=1)
callback = SaveOnBestTrainingRewardCallback(check_freq=100, save_path=log_dir)

# Train the agent
timesteps = 2e4
model.learn(total_timesteps=int(timesteps), callback=callback)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Gripper Closing Environment")
plt.show()