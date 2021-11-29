import numpy as np

from enum import Enum
from tiago_rl.envs import BulletRobotEnv

class ObsConfig(Enum):
    GOAL_DELTA = 1
    POSITIONS = 2
    VELOCITIES = 3

class GripperCloseEnv(BulletRobotEnv):

    def __init__(self, obs_config: list[ObsConfig] = [ObsConfig.GOAL_DELTA], *args, **kwargs) -> None:
        self.q_goal = 0.0
        self.joint_range = [0.001, 0.043]

        self.obs_config = obs_config

        joints = [
            'gripper_right_finger_joint',
            'gripper_left_finger_joint',
        ]

        initial_state = [
            0.043,
            0.043,
        ]

        max_vel = 0.02
        max_joint_velocities = {
            'gripper_right_finger_joint': max_vel,
            'gripper_left_finger_joint': max_vel,
        }

        BulletRobotEnv.__init__(self,
                                joints=joints,
                                initial_state=initial_state,
                                max_joint_velocities=max_joint_velocities,
                                cam_yaw=120.5228271484375,
                                cam_pitch=-68.42454528808594,
                                cam_distance=1.1823151111602783,
                                cam_target_position=(-0.2751278877258301, -0.15310688316822052, -0.27969369292259216),
                                robot_model="gripper_tactile.urdf",
                                robot_pos=[0.0, 0.0, 0.265],
                                *args, 
                                **kwargs)

    def _goal_delta(self, q):
        return q - self.q_goal

    def _get_obs(self):
        """
        construct observation space:
        - q
        - q_dot
        - \Delta q_goal
        """

        _obs = []
        if ObsConfig.GOAL_DELTA in self.obs_config:
            _obs.append(self._goal_delta(self.current_pos))
        if ObsConfig.POSITIONS in self.obs_config:
            _obs.append(self.current_pos)
        if ObsConfig.VELOCITIES in self.obs_config:
            _obs.append(self.current_vel)

        return np.concatenate(_obs)

    def _compute_reward(self):
        """
        """
        return -np.sum(np.abs(self._goal_delta(self.current_pos)))

    def _reset_callback(self):
        """
        sample new goal and new initial state
        """
        initial_state = np.round(np.array(2*[np.random.uniform(*self.joint_range)]), 4)
        self.initial_state = list(zip(self.joints, initial_state))
        
        self.q_goal = np.round(np.random.uniform(*self.joint_range), 4)

    
    def _is_success(self):
        """
        not sure, maybe return True if current value is in some confidence interval
        """
        return False