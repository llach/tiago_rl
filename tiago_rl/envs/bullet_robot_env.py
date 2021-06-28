import os
import time
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

DEFAULT_SIZE = 500

class BulletRobotEnv(gym.Env):
    """
    Superclass for PyBullet-based gym environments for TIAGo.
    This code's starting point was the MuJoCo-based robotics environment from gym:
    https://github.com/openai/gym/blob/master/gym/envs/robotics/robot_env.py
    """
    def __init__(self, initial_state, n_actions, dt=1./240., render=False):

        if render:
            self.client_id = p.connect(p.SHARED_MEMORY)

            if self.client_id < 0:
                self.client_id = p.connect(p.GUI)

            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        else:
            self.client_id = p.connect(p.DIRECT)

        self.dt = dt

        # will be set by child environment
        self.robotId = None  # sim ID of robot model
        self.jn2Idx = None  # dict of joint names to model indices

        self.seed()
        self._env_setup(initial_state=initial_state)
        self.initial_state = initial_state

        # self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._step_sim()
        self._step_callback()

        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(),
        }
        reward = self._compute_reward()
        return obs, reward, done, info

    def reset(self):
        # For now, we leave the initial reset loop intact. Might come in handy
        # once we add object location randomization or other features that could
        # result in bad initial states
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        # self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def render(self, width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        # TODO render RGB image and return it
        return np.array([])

    # Extension methods
    # ----------------------------

    def _load_model(self):
        # load PyBullet default plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.01])

        # set asset directory as search path for child environments
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), '../assets', ))

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        """

        # reset bullet
        p.resetSimulation()

        # reset gravity
        p.setGravity(0, 0, -9.8)

        self._load_model()
        self._env_setup(self.initial_state)

        # step a few times for things to get settled
        for _ in range(100):
            self._step_sim()

        return True

    def _set_state(self, state):
        if self.robotId:
            for jn, q in state:
                self._set_joint_pos(self.jn2Idx[jn], q)

    def _set_joint_pos(self, joint_idx, joint_pos):
        if self.robotId:
            p.resetJointState(self.robotId, joint_idx, joint_pos)

    def _env_setup(self, initial_state):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        self._set_state(initial_state)

    def _step_sim(self):
        """Steps one timestep.
        """
        p.stepSimulation()
        time.sleep(self.dt)

    def _compute_reward(self):
        """Returns the reward for this timestep.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass