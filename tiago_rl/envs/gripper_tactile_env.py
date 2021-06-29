import numpy as np
import pybullet as p

from collections import deque

from tiago_rl.envs import BulletRobotEnv
from tiago_rl.envs.utils import link_to_idx, joint_to_idx


class GripperTactileEnv(BulletRobotEnv):

    def __init__(self, initial_state=None, dt=1./240., render=False, force_noise_mu=0.0, force_noise_sigma=1.0, force_smoothing=4):
        self.objectId = None

        self.force_smoothing = force_smoothing
        self.force_noise_mu = force_noise_mu
        self.force_noise_sigma = force_noise_sigma

        self.force_buffer_r = deque(maxlen=self.force_smoothing)
        self.force_buffer_l = deque(maxlen=self.force_smoothing)

        BulletRobotEnv.__init__(self,
                                dt=dt,
                                n_actions=2, # 3 if torso is included
                                render=render,
                                joints=['gripper_right_finger_joint', 'gripper_left_finger_joint'], # , 'torso_to_arm']
                                initial_state=initial_state or [0.045, 0.045])

        if render:
            # focus grasping scene
            p.resetDebugVisualizerCamera(1.1823151111602783, 120.5228271484375, -68.42454528808594,
                                         (-0.2751278877258301, -0.15310688316822052, -0.27969369292259216))

        self.reset()

    # BulletRobotEnv methods
    # ----------------------------

    def _load_model(self):
        super()._load_model()

        # load scene objects
        self.objectId = p.loadURDF("objects/object.urdf", basePosition=[0.04, 0.02, 0.6])
        self.robotId = p.loadURDF("gripper_tactile.urdf", basePosition=[0.0, 0.0, 0.27])

        # link name to link index mappings
        self.robot_link_to_index = link_to_idx(self.robotId)
        self.object_link_to_index = link_to_idx(self.objectId)

        # joint name to joint index mapping
        self.jn2Idx = joint_to_idx(self.robotId)

    def _compute_reward(self):
        return 0

    def _get_obs(self):
        # TODO add gripper position and velocity to observation space
        pos, vel = self._get_joint_states()

        if not self.objectId:
            forces = [0.0, 0.0]
        else:
            # get current forces
            self.force_buffer_r.append(self._get_contact_force(self.robotId, self.objectId,
                                       self.robot_link_to_index['gripper_right_finger'],
                                       self.object_link_to_index['objectLink']))
            self.force_buffer_l.append(self._get_contact_force(self.robotId, self.objectId,
                                       self.robot_link_to_index['gripper_left_finger'],
                                       self.object_link_to_index['objectLink']))

            # create force array
            forces = np.array([
                np.mean(self.force_buffer_r),
                np.mean(self.force_buffer_l)
            ])

        obs = np.concatenate([pos, vel, forces])
        return {
            'observation': obs
        }

    def _is_success(self):
        return False

    def _transform_forces(self, force):
        return (force / 100) + np.random.normal(self.force_noise_mu, self.force_noise_sigma)