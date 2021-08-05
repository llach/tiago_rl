import os
import time
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    Modified version of an example from the stable baselines 2 docs.

    :param check_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param mean_n: (int) number of episodes the mean will be calculated on
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, save_path: str, mean_n=100, verbose=1,
                 periodic_saving: int = 0, periodic_saving_offset: int = 0, total_steps: int = 0):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.mean_n = mean_n
        self.start_time = time.time()
        self.ps = periodic_saving
        self.ps_off = periodic_saving_offset
        self.total_steps = total_steps

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.ps > 0 and self.n_calls >= self.ps_off and self.n_calls % self.ps == 0:
            print(f"Saving checkpoint model at {self.n_calls}")
            self.model.save(f'{self.save_path}/model{int(self.n_calls/1000)}')

        if self.n_calls % self.check_freq == 0:

            # load training reward
            x, y = ts2xy(load_results(self.save_path), 'timesteps')
            if len(x) > 0:
                # mean training reward over the last N episodes
                mean_reward = np.mean(y[-self.mean_n:])

                # new best model, thus we save it
                if mean_reward > self.best_mean_reward:
                    print("!!! new best mean reward {:.2f} !!! before: {:.2f}".format(mean_reward, self.best_mean_reward))
                    self.best_mean_reward = mean_reward
                    print("Saving new best model to {}".format(self.save_path))
                    self.model.save(f'{self.save_path}/best_model')
                else:
                    print("Best mean reward was: {:.2f}".format(self.best_mean_reward))

                if self.total_steps > 0:
                    time_elapsed = time.time() - self.start_time
                    fps = int(self.num_timesteps / (time_elapsed + 1e-8))
                    eta_m = int((self.total_steps - self.n_calls) / fps / 60)

                    print(f"ETA {int(eta_m / 60)}:{eta_m % 60:02d} ||  FPS {fps}")
        return True