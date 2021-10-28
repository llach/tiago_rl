import os
import sys
import time
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

from tiago_rl.envs.utils import link_to_idx, joint_to_idx

DEFAULT_SIZE = 500
POS_CTRL = 'pos'
VEL_CTRL = 'vel'


class BulletRobotEnv(gym.Env):
    """
    Superclass for PyBullet-based gym environments for TIAGo.
    This code's starting point was the MuJoCo-based robotics environment from gym:
    https://github.com/openai/gym/blob/master/gym/envs/robotics/robot_env.py
    """
    def __init__(self, initial_state, joints, n_actions=None, show_gui=False, control_mode='vel',
                 cam_distance=None, cam_yaw=None, cam_pitch=None, cam_target_position=None, max_joint_velocities=None,
                 robot_model=None, robot_pos=None, object_model=None, object_pos=None, table_model=None, table_pos=None):

        # PyBullet camera settings for visualisation and RGB array rendering
        self.cam_distance = cam_distance or 1.5
        self.cam_yaw = cam_yaw or 180.0
        self.cam_pitch = cam_pitch or -40.0
        self.cam_target_position = cam_target_position or (0.0, 0.0, 0.0)

        self.control_mode = control_mode
        assert self.control_mode in {POS_CTRL, VEL_CTRL}, f"unknown control mode {self.control_mode}"

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

        self.dt = 1./240. # don't change; pybullet is tuned for this value
        self.joints = joints
        self.num_joints = len(joints)
        self.initial_state = list(zip(self.joints, initial_state))
        self.max_joint_velocities = max_joint_velocities
        self.n_actions = n_actions or len(joints)

        self.total_max_vel = np.sum(np.abs(list(self.max_joint_velocities.values())))

        # current state
        self.current_pos = initial_state
        self.current_vel = np.array(self.num_joints*[0.0])
        self.current_acc = np.array(self.num_joints*[0.0])

        self.last_pos = initial_state
        self.last_vel = np.array(self.num_joints*[0.0])
        self.last_acc = np.array(self.num_joints*[0.0])

        self.desired_action = np.array(self.num_joints*[0.0])
        self.desired_action_clip = np.array(self.num_joints*[0.0])

        self.uli = 0.045
        self.lli = 0.0

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

        if self.control_mode == POS_CTRL:
            action_high = 10
        elif self.control_mode == VEL_CTRL:
            action_high = 1
        self.action_space = spaces.Box(-action_high, action_high, shape=(self.n_actions,), dtype='float32')

        high = 5
        self.observation_space = spaces.Box(-high, high, shape=obs.shape, dtype='float32')

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.last_pos = self.current_pos.copy()
        self.last_vel = self.current_vel.copy()
        self.last_acc = self.current_acc.copy()

        if np.any(np.isnan(action)):
            print(f"action has NaNs: {action}")
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self._set_action(action)

        self._step_sim()
        self._step_callback()

        self.current_pos, self.current_vel = self._get_joint_states()
        self.current_acc = (self.last_vel-self.current_vel)

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

        self._reset_callback()
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
        p.setAdditionalSearchPath(os.path.join(sys.modules['tiago_rl'].__path__[0], 'assets', ))

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

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        self.desired_action = action.copy()

        # clipping actions is encouraged in the pybullet docs even though we have limits set in the simulation
        self._clip_actions()

        if self.joints:
            for i, [jn, act] in enumerate(zip(self.joints, self.desired_action_clip)):
                ji = self.jn2Idx[jn]
                if self.control_mode == POS_CTRL:
                    p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                            jointIndex=ji,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=act)
                elif self.control_mode == VEL_CTRL:
                    p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                            jointIndex=ji,
                                            controlMode=p.VELOCITY_CONTROL,
                                            targetVelocity=act)
        else:
            print("Environment has no joints specified!")

    def _clip_actions(self):
        if self.control_mode == POS_CTRL:
            # this clipping is not velocity-sensitive as we don't want to mess with the internal PI controller of bullet
            # previous experiments have proven for it to not reach the target if too small position deltas are set
            self.desired_action_clip = np.clip(self.desired_action, self.lli, self.uli)

        elif self.control_mode == VEL_CTRL:

            # this clips velocities only and ignores current position
            for i, [jn, act] in enumerate(zip(self.joints, self.desired_action)):
                if jn in self.max_joint_velocities:
                    mv = self.max_joint_velocities[jn]
                    self.desired_action_clip[i] = np.clip(act, -mv, mv)

                else:
                    print(f"no velocity limit given for joint {jn}. won't clip.")

    def _get_obs(self):
        """Returns the observation.
        """
        pos, vel = self._get_joint_states()
        return np.concatenate([pos, vel])

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

    def _reset_callback(self):
        """A custom callback that is called after resetting the simulation. Can be used
        to randomize certain environment properties.
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

            if self.max_joint_velocities:
                for j, max_vel in self.max_joint_velocities.items():
                    p.changeDynamics(bodyUniqueId=self.robotId,
                                     linkIndex=self.jn2Idx[j],
                                     maxJointVelocity=max_vel,
                                     jointLowerLimit=0.0,
                                     jointUpperLimit=0.045)

    def _set_joint_pos(self, joint_idx, joint_pos):
        if self.robotId:
            p.resetJointState(self.robotId, joint_idx, joint_pos)

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
            return self._transform_forces(f_raw), f_raw > 0.0
        else:
            return 0.0, False

    def _get_joint_states(self):
        pos = []
        vel = []

        for js in p.getJointStates(self.robotId, [self.jn2Idx[jn] for jn in self.joints]):
            pos.append(js[0])
            vel.append(js[1])
        return np.array(pos), np.array(vel)

    def create_desired_state(self, des_qs):
        """Creates a complete desired joint state from a partial one.
        """
        ds = self.desired_action.copy()

        for jn, des_q in des_qs.items():
            if jn not in self.joints:
                print(f"Unknown joint {jn}")
                exit(-1)
            ds[self.joints.index(jn)] = des_q
        return ds

    def get_state_dicts(self):
        return dict(zip(self.joints, self.current_pos)), dict(zip(self.joints, self.current_vel))

    def get_desired_q_dict(self):
        return dict(zip(self.joints, self.desired_action))

    def get_object_velocity(self):
        if self.objectId:
            return p.getBaseVelocity(self.objectId)
