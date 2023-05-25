import os
import numpy as np

from tiago_rl.envs import GripperPosEnv
from tiago_rl.models import PosModel
from tiago_rl.misc import PosVis

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt


def deterministic_eval(env, model, vis, goals):
    rewards = []
    for i in range(len(goals)):
        obs, _ = env.reset()
        env.set_goal(goals[i])

        if vis: vis.reset()
        done = False
        cumrew = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            cumrew += reward

            if vis: vis.update_plot(action=action, reward=reward)
        rewards.append(cumrew)
    return rewards

with_vis = 0

# Environment setup
# ----------------------------
env = GripperPosEnv(**{"render_mode": "human"} if with_vis else {})
env = TimeLimit(env, max_episode_steps=100)
vis = PosVis(env) if with_vis else None

agent = PPO('MlpPolicy', env, verbose=1)
agent = agent.load("/tmp/continuous/best_model.zip")

pos_model = PosModel()

goals = np.round(np.linspace(0.0, 0.045, 50), 4)
rl_rewards = deterministic_eval(env, agent, vis, goals)
oracle_rewards = deterministic_eval(env, pos_model, vis, goals)

print(f"RL {np.mean(rl_rewards)}±{np.std(rl_rewards)}")
print(f"OR {np.mean(oracle_rewards)}±{np.std(oracle_rewards)}")

plt.title("GripperPosEnv - continuous r(t)")
plt.xlabel("goal position")
plt.ylabel("cumulative episode reward")

plt.scatter(goals, rl_rewards, label="π")
plt.scatter(goals, oracle_rewards, label="Oracle")

plt.ylim(0, 1.1*np.max([rl_rewards, oracle_rewards]))
plt.legend()
plt.tight_layout()
plt.show()