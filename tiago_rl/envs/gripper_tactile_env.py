import numpy as np
from collections import deque

from tiago_rl.envs import BulletRobotEnv


class LoadCellTactileEnv(BulletRobotEnv):

    def __init__(self, joints, initial_state=None, dt=1./240., show_gui=False, force_noise_mu=0.0, force_noise_sigma=0.0077,
                 force_smoothing=4, cam_distance=None, cam_yaw=None, cam_pitch=None, cam_target_position=None,
                 robot_model=None, robot_pos=None, object_model=None, object_pos=None, table_model=None, table_pos=None):

        self.force_smoothing = force_smoothing
        self.force_noise_mu = force_noise_mu
        self.force_noise_sigma = force_noise_sigma

        self.force_buffer_r = deque(maxlen=self.force_smoothing)
        self.force_buffer_l = deque(maxlen=self.force_smoothing)

        BulletRobotEnv.__init__(self,
                                dt=dt,
                                joints=joints,
                                show_gui=show_gui,
                                n_actions=len(joints),
                                initial_state=initial_state,
                                cam_yaw=cam_yaw,
                                cam_pitch=cam_pitch,
                                cam_distance=cam_distance,
                                cam_target_position=cam_target_position,
                                robot_model=robot_model,
                                robot_pos=robot_pos,
                                object_model=object_model,
                                object_pos=object_pos,
                                table_model=table_model,
                                table_pos=table_pos)

    # BulletRobotEnv methods
    # ----------------------------

    def _transform_forces(self, force):
        return (force / 100) + np.random.normal(self.force_noise_mu, self.force_noise_sigma)

    def _get_obs(self):
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

    def _compute_reward(self):
        # todo add reward calculation
        return 0


class GripperTactileEnv(LoadCellTactileEnv):

    def __init__(self, initial_state=None, dt=1./240., show_gui=False, force_noise_mu=0.0, force_noise_sigma=1.0, force_smoothing=4):

        LoadCellTactileEnv.__init__(self,
                                    dt=dt,
                                    show_gui=show_gui,
                                    joints=['gripper_right_finger_joint', 'gripper_left_finger_joint'], # , 'torso_to_arm']
                                    initial_state=initial_state or [0.045, 0.045],
                                    cam_yaw=120.5228271484375,
                                    cam_pitch=-68.42454528808594,
                                    cam_distance=1.1823151111602783,
                                    cam_target_position=(-0.2751278877258301, -0.15310688316822052, -0.27969369292259216),
                                    force_noise_mu=force_noise_mu,
                                    force_noise_sigma=force_noise_sigma,
                                    force_smoothing=force_smoothing,
                                    robot_model="gripper_tactile.urdf",
                                    robot_pos=[0.0, 0.0, 0.27],
                                    object_model="objects/object.urdf",
                                    object_pos=[0.04, 0.02, 0.6])