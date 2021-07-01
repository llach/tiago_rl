import os
import time
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

from tiago_rl.envs.utils import link_to_idx, joint_to_idx

DEFAULT_SIZE = 500


class BulletRobotEnv(gym.Env):
    """
    Superclass for PyBullet-based gym environments for TIAGo.
    This code's starting point was the MuJoCo-based robotics environment from gym:
    https://github.com/openai/gym/blob/master/gym/envs/robotics/robot_env.py
    """
    def __init__(self, initial_state, joints, n_actions=None, dt=1./240., show_gui=False,
                 cam_distance=None, cam_yaw=None, cam_pitch=None, cam_target_position=None,
                 robot_model=None, robot_pos=None, object_model=None, object_pos=None, table_model=None, table_pos=None):

        # PyBullet camera settings for visualisation and RGB array rendering
        self.cam_distance = cam_distance or 1.5
        self.cam_yaw = cam_yaw or 180.0
        self.cam_pitch = cam_pitch or -40.0
        self.cam_target_position = cam_target_position or (0.0, 0.0, 0.0)

        if show_gui:
            self.client_id = p.connect(p.SHARED_MEMORY)

            if self.client_id < 0:
                self.client_id = p.connect(p.GUI)

            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

            # set camera view
            p.resetDebugVisualizerCamera(self.cam_distance,
                                         self.cam_yaw,
                                         self.cam_pitch,
                                         self.cam_target_position)
        else:
            self.client_id = p.connect(p.DIRECT)
        self.show_gui = show_gui

        self.dt = dt
        self.joints = joints
        self.num_joints = len(joints)
        self.initial_state = list(zip(self.joints, initial_state))
        self.n_actions = n_actions or len(joints)

        # current state
        self.current_pos = initial_state
        self.current_vel = self.num_joints*[0.0]

        self.desired_pos = self.current_pos

        # needs to be set by child environment
        self.robotId = None  # sim ID of robot model
        self.jn2Idx = None  # dict of joint names to model indices
        self.objectId = None # sim ID of grasp object

        self.robot_model = robot_model
        self.robot_pos = robot_pos or [0, 0, 0]
        self.object_model = object_model
        self.object_pos = object_pos or [0, 0, 0]
        self.table_model = table_model
        self.table_pos = table_pos or [0, 0, 0]

        self.seed()
        obs = self.reset()

        self.action_space = spaces.Box(-np.inf, np.inf, shape=(self.n_actions,), dtype='float32')
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
        reward = self._compute_reward()
        done = False
        info = {
            'is_success': self._is_success(), # used by the HER algorithm
        }
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
        """Render RGB Array from scene. Based on:
        https://github.com/robotology-playground/pybullet-robot-envs/blob/master/pybullet_robot_envs/envs/icub_envs/icub_reach_gym_env.py#L267
        """

        view_matrix = p.computeViewMatrixFromYawPitchRoll(roll=0,
                                                          yaw=self.cam_yaw,
                                                          pitch=self.cam_pitch,
                                                          distance=self.cam_distance,
                                                          upAxisIndex=2,
                                                          cameraTargetPosition=self.cam_target_position)

        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   nearVal=0.1,
                                                   farVal=100.0,
                                                   aspect=float(width) / height)

        (_, _, px, _, _) = p.getCameraImage(width=width, height=height,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        """Cleanup sim.
        """
        p.disconnect()

    # Extension methods
    # ----------------------------

    def _load_model(self):
        # load PyBullet default plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.01])

        # set asset directory to local asset directory
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), '../assets', ))

        if not self.robot_model:
            print("Robot model path missing!")
            exit(-1)

        self.robotId = p.loadURDF(self.robot_model, basePosition=self.robot_pos)

        # link name to link index mappings
        self.robot_link_to_index = link_to_idx(self.robotId)

        # joint name to joint index mapping
        self.jn2Idx = joint_to_idx(self.robotId)

        if self.object_model:
            # load grasping object
            self.objectId = p.loadURDF(self.object_model, basePosition=self.object_pos)
            self.object_link_to_index = link_to_idx(self.objectId)

        if self.table_model:
            # load table
            self.tableId = p.loadURDF(self.table_model, basePosition=self.table_pos)
            self.table_link_to_index = link_to_idx(self.tableId)

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

    def _env_setup(self, initial_state):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        self._reset_state(initial_state)

    def _step_sim(self):
        """Steps one timestep.
        """
        p.stepSimulation()
        time.sleep(self.dt)

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        self.desired_pos = action

        if self.joints:
            for jn, des_q in zip(self.joints, action):
                self._set_desired_q(self.jn2Idx[jn], des_q)
        else:
            print("Environment has no joints specified!")

    def _get_obs(self):
        """Returns the observation.
        """
        pos, vel = self._get_joint_states()
        return {
            'observation': np.concatenate([pos, vel])
        }

    def _compute_reward(self):
        """Returns the reward for this timestep.
        """
        raise NotImplementedError()

    def _is_success(self):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _transform_forces(self, force):
        """Transformations to forces can be applied here (e.g. add noise,
        """
        return force

    # PyBullet Wrapper
    # ----------------------------

    def _reset_state(self, state):
        if self.robotId:
            for jn, q in state:
                self._set_joint_pos(self.jn2Idx[jn], q)

    def _set_joint_pos(self, joint_idx, joint_pos):
        if self.robotId:
            p.resetJointState(self.robotId, joint_idx, joint_pos)

    def _set_desired_q(self, joint_idx, des_q):
        p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                jointIndex=joint_idx,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=des_q)

    def _calculate_force(self, contacts):
        if not contacts:
            return 0.0

        # we might want to do impose additional checks upon contacts
        f = [c[9] for c in contacts]
        return np.sum(f)

    def _get_contact_force(self, bodyA, bodyB, linkA, linkB):
        if self.robotId:
            cps = p.getContactPoints(bodyA=bodyA,
                                     bodyB=bodyB,
                                     linkIndexA=linkA,
                                     linkIndexB=linkB)
            f_raw = self._calculate_force(cps)
            return self._transform_forces(f_raw)
        else:
            return 0.0

    def _get_joint_states(self):
        pos = []
        vel = []

        for js in p.getJointStates(self.robotId, [self.jn2Idx[jn] for jn in self.joints]):
            pos.append(js[0])
            vel.append(js[1])
        return pos, vel

    def create_desired_state(self, des_qs):
        """Creates a complete desired joint state from a partial one.
        """
        ds = self.desired_pos.copy()

        for jn, des_q in des_qs.items():
            if jn not in self.joints:
                print(f"Unknown joint {jn}")
                exit(-1)
            ds[self.joints.index(jn)] = des_q
        return ds

