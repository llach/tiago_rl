import numpy as np
import pybullet as p

from tiago_rl.envs import BulletRobotEnv
from tiago_rl.envs.utils import link_to_idx, joint_to_idx


class GripperTactileEnv(BulletRobotEnv):

    def __init__(self, initial_state=None, dt=1./240., render=False):
        initial_state = initial_state or [
            ['gripper_right_finger_joint', 0.045],
            ['gripper_left_finger_joint', 0.045],
            ['torso_to_arm', 0.00]
        ]

        BulletRobotEnv.__init__(self,
                                dt=dt,
                                n_actions=2, # 3 if torso is included
                                render=render,
                                initial_state=initial_state)

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
        obs = np.array([0])

        return {
            'observation': obs
        }

    def _set_action(self, action):
        pass

    def _is_success(self):
        return False