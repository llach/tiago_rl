import numpy as np

from enum import Enum
from tiago_rl.envs import BulletRobotEnv

class ObsConfig(str, Enum):
    GOAL_DELTA = "goal_delta"
    POSITIONS = "positions"
    VELOCITIES = "velocities"

def map_in_range(v: float, vrange: list, trange: list):
    return (v-vrange[0])*((trange[1]-trange[0])/(vrange[1]-vrange[0]))+trange[0]

class GripperCloseEnv(BulletRobotEnv):

    def __init__(self, obs_config: list[ObsConfig] = [ObsConfig.GOAL_DELTA], sample_initial: bool = False, sample_goal: bool = False, *args, **kwargs) -> None:
        self.q_goal = 0.0
        self.joint_range = [0.001, 0.043]
        self.pos_range = [0.0, 0.043]

        self.obs_config = obs_config
        self.sample_initial = sample_initial
        self.sample_goal = sample_goal

        joints = [
            'gripper_right_finger_joint',
            'gripper_left_finger_joint',
        ]

        initial_state = [
            0.043,
            0.043,
        ]

        max_vel = 0.08
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

    def _match_obs(self, oc: ObsConfig):
        if oc == ObsConfig.GOAL_DELTA:
            return self._goal_delta(self.current_pos)
        if oc == ObsConfig.POSITIONS:
            return self.current_pos
        if oc == ObsConfig.VELOCITIES:
            return self.current_vel
        else:
            print("unknown obs type {oc}")
            exit(-1)

    def _get_obs(self):
        """
        construct observation space:
        - q
        - q_dot
        - \Delta q_goal
        """

        _obs = []
        for oc in self.obs_config:
            _obs.append(self._match_obs(oc))
        return np.concatenate(_obs)

    def _compute_reward(self):
        gd = np.abs(self._goal_delta(self.current_pos))
        return -np.sum(map_in_range(gd, self.pos_range, [0.0, 1.0]))

    def _reset_callback(self):
        """
        sample new goal and new initial state
        """
        if self.sample_goal:
            self.q_goal = np.round(np.random.uniform(*self.joint_range), 4)
        else:
            self.q_goal = 0.0
        
        if self.sample_initial:
            initial_state = np.round(np.array(2*[np.random.uniform(*self.joint_range)]), 4)
        else:
            initial_state = np.array(2*[0.043])
        self.initial_state = list(zip(self.joints, initial_state))

        self.pos_range = [min(self.q_goal, initial_state[0]), max(self.q_goal, initial_state[0])]
    
    def _is_success(self):
        """
        not sure, maybe return True if current value is in some confidence interval
        """
        return False

if __name__ == "__main__":
    print(map_in_range(0.02, [0.01, 0.03], [0.0, 1.0]))
    pass