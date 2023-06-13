import os
import numpy as np

from tiago_rl.envs import GripperTactileEnv
from tiago_rl.models import ForcePI
from tiago_rl.misc import TactileVis

from stable_baselines3 import PPO, TD3
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt


def deterministic_eval(env, model, vis, goals, reset_cb=None):
    rewards = []
    for i in range(len(goals)):
        obs, _ = env.reset()
        env.set_goal(goals[i])

        if vis: vis.reset()
        done = False
        cumrew = 0

        if reset_cb: reset_cb(env, model, goals[i])

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            cumrew += reward

            if vis: vis.update_plot(action=action, reward=reward)
        print(cumrew)
        rewards.append(cumrew)
    return rewards

def force_reset_cb(env, model, goal): model.reset(goal)

with_vis = 1

# Environment setup
# ----------------------------
env = GripperTactileEnv(alpha=2, delta=2, **{"render_mode": "human"} if with_vis else {})
env = TimeLimit(env, max_episode_steps=250)
vis = TactileVis(env) if with_vis else None

agent = TD3('MlpPolicy', env, verbose=1)
agent = agent.load("/tmp/tactile/best_model")

fc = ForcePI(env.dt, env.fmax, env.fgoal, env.ftheta, Kp=1.5, Ki=3.1, k=160)

goals = np.round(np.linspace(0.3, 0.6, 10), 4)[::-1]
# goals = 10*[0.6]
rl_rewards = deterministic_eval(env, agent, vis, goals)
oracle_rewards = deterministic_eval(env, fc, vis, goals, reset_cb=force_reset_cb)

print(f"RL {np.mean(rl_rewards)}±{np.std(rl_rewards)}")
print(f"OR {np.mean(oracle_rewards)}±{np.std(oracle_rewards)}")

print(np.std(rl_rewards)/np.mean(rl_rewards))

plt.title("GripperTactileEnv")
plt.xlabel("goal force")
plt.ylabel("cumulative episode reward")

plt.scatter(goals, rl_rewards, label=f"π {np.mean(rl_rewards):.0f}±{np.std(rl_rewards):.1f}")
plt.scatter(goals, oracle_rewards, label=f"Oracle{np.mean(oracle_rewards):.0f}±{np.std(oracle_rewards):.1f}")

# plt.ylim(0, 1.1*np.max([rl_rewards, oracle_rewards]))
# plt.ylim(0, 1.1*np.max([oracle_rewards]))
plt.legend()
plt.tight_layout()
plt.show()