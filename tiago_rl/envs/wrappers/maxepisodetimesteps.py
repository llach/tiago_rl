import gym
from gym import logger


class MaxEpisodeTimestepWrapper(gym.Wrapper):

    def __init__(self, env, max_timesteps):
        self.max_timesteps = max_timesteps
        super(MaxEpisodeTimestepWrapper, self).__init__(env)

        self.current_t = 0
        self.steps_since_done = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.current_t += 1
        done = self.current_t > self.max_timesteps or done

        if done:
            if not self.steps_since_done:
                self.steps_since_done = 0
            elif self.steps_since_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.current_t = 0
        self.steps_since_done = None

        self.env.reset(**kwargs)
