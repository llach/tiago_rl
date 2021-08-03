import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import TimeLimit


class EvalCallback(BaseCallback):

    def __init__(self, n_steps: int, save_path: str, eval_env: TimeLimit, iterations: int = 5, verbose: int = 0):
        super(EvalCallback, self).__init__(verbose=verbose)
        self.iterations = iterations
        self.save_path = save_path
        self.eval_env = eval_env
        self.n_steps = n_steps

        filename = f"{save_path}/eval.csv"
        self.file_handler = open(filename, "wt")
        self.file_handler.write('timestep;rewards\n')
        self.file_handler.flush()

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.n_steps == 0:
            print(f"Evaluating environment {self.iterations} times ...")
            rewards = self._run_eval()
            self._write_rewards(rewards)
            print(np.mean(rewards), np.std(rewards), rewards)

        return True

    def _write_rewards(self, rs):
        self.file_handler.write(f'{self.n_calls};{rs}\n')
        self.file_handler.flush()

    def _run_eval(self):
        rs = []
        for _ in range(self.iterations):
            obs = self.eval_env.reset()
            rewards = []
            for i in range(self.eval_env._max_episode_steps):
                action = self.model.predict(obs, deterministic=True)[0]
                obs, reward, done, info = self.eval_env.step(action=action)
                rewards.append(reward)
            rs.append(np.sum(rewards))
        return rs
