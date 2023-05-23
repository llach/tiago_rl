import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

from tiago_rl.envs import GripperTactileEnv
from tiago_experiments import SaveOnBestTrainingRewardCallback

from gymnasium.wrappers import TimeLimit, FrameStack, FlattenObservation

import torch as th
import torch.nn.functional as F
from torch import nn

# Create log dir
timesteps = 2e7
log_dir = "/tmp/"
os.makedirs(log_dir, exist_ok=True)

# Environment setup
# ----------------------------
env = GripperTactileEnv()
env = TimeLimit(env, max_episode_steps=300)
# env = FrameStack(env=env, num_stack=5, lz4_compress=False)
# env = FlattenObservation(env=env)
env = Monitor(env, log_dir)

model = PPO('MlpPolicy', env, verbose=1,)
            # policy_kwargs=dict(
            #     share_features_extractor=True,
            #     net_arch=dict(pi=[64, 64], vf=[64, 128]),
            #     activation_fn=nn.Tanh,
            #     squash_output=True
            # ))
callback = SaveOnBestTrainingRewardCallback(env=env,
                                            check_freq=2000,
                                            total_steps=timesteps,
                                            save_path=log_dir,
                                            step_offset=0,
                                            mean_n=20
                                            )

# Train the agent
model.learn(total_timesteps=int(timesteps), callback=callback)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Gripper Closing Environment")
plt.show()