import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import TimeLimit


class EvalCallback(BaseCallback):

    def __init__(self, n_steps: int, save_path: str, eval_env: TimeLimit,
                 iterations: int = 5, verbose: int = 0, plot: bool = True):
        super(EvalCallback, self).__init__(verbose=verbose)
        self.iterations = iterations
        self.save_path = save_path
        self.eval_env = eval_env
        self.n_steps = n_steps

        filename = f"{save_path}/eval.csv"
        self.file_handler = open(filename, "wt")
        self.file_handler.write('timestep;rewards\n')
        self.file_handler.flush()

        self.plot_path = f"{save_path}/plot" if plot else None

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.plot_path:
            os.makedirs(self.plot_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.n_steps == 0:
            print(f"Evaluating environment {self.iterations} times @ {self.n_calls}")
            rewards = self._run_eval()
            self._write_rewards(rewards)
            print(np.mean(rewards), np.std(rewards), rewards)
        return True

    def _write_rewards(self, rs):
        self.file_handler.write(f'{self.n_calls};{rs}\n')
        self.file_handler.flush()

    def _run_eval(self):
        rs = []
        for it in range(self.iterations):
            self.eval_env.seed(self.n_steps+it)
            obs = self.eval_env.reset()

            rewards = []
            forces = [[], []]
            vr = [[], []]
            vl = [[], []]

            for i in range(self.eval_env._max_episode_steps):
                action = self.model.predict(obs, deterministic=True)[0]

                obs, reward, done, info = self.eval_env.step(action=action)

                forces[0].append(self.eval_env.current_forces[0])
                forces[1].append(self.eval_env.current_forces[1])

                _, jv = self.eval_env.get_state_dicts()
                dv = self.eval_env.get_desired_q_dict()

                # actual, desired
                vr[0].append((jv['gripper_right_finger_joint']))
                vr[1].append((dv['gripper_right_finger_joint']))

                vl[0].append((jv['gripper_left_finger_joint']))
                vl[1].append((dv['gripper_left_finger_joint']))

                rewards.append(reward)

            self._plot(forces, rewards, vr, vl, it)
            rs.append(np.sum(rewards))
        return rs

    def _plot(self, forces, rewards, vr, vl, it):
        plot_fn = f"{self.n_calls:.0e}-{it}.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(forces[0], label="right", c="green")
        axes[0, 0].plot(forces[1], label="left", c="red")
        axes[0, 0].axhline(self.eval_env.force_threshold, c="silver")
        axes[0, 0].axhline(self.eval_env.target_forces[0], c="dimgrey")

        axes[0, 1].plot(np.cumsum(rewards), label='reward', c="purple")

        axes[1, 0].plot(vr[0], label="act R", c="green")
        axes[1, 0].plot(vr[1], label="des R", c="limegreen")

        axes[1, 1].plot(vl[0], label="act L", c="red")
        axes[1, 1].plot(vl[1], label="des L", c="tomato")

        for ax in axes.flatten():
            ax.legend()

        fig.suptitle(f"Evaluation {self.n_calls:.0e} - {it}", fontsize="15", fontweight="semibold")

        fig.tight_layout()
        plt.savefig(f"{self.plot_path}/{plot_fn}")
        # plt.show()

        plt.clf()
