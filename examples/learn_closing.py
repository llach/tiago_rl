import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

from tiago_rl.envs.gripper_env import GripperEnv
from tiago_experiments import SaveOnBestTrainingRewardCallback

from gymnasium.wrappers import TimeLimit

# Create log dir
timesteps = 1e6
log_dir = "/tmp/"
os.makedirs(log_dir, exist_ok=True)

# Environment setup
# ----------------------------
env = GripperEnv()
env = TimeLimit(env, max_episode_steps=500)
env = Monitor(env, log_dir)

model = PPO('MlpPolicy', env, verbose=1)
callback = SaveOnBestTrainingRewardCallback(env=env,
                                            check_freq=2000,
                                            total_steps=2e4,
                                            save_path=log_dir,
                                            step_offset=0,
                                            mean_n=20
                                            )

# Train the agent
model.learn(total_timesteps=int(timesteps), callback=callback)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Gripper Closing Environment")
plt.show()