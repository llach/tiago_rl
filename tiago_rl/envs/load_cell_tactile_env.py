import numpy as np
from collections import deque

from tiago_rl.envs import BulletRobotEnv


def force_delta(force_a, force_b):
    assert force_a.shape == force_b.shape
    return force_a - force_b


def map_in_range(v, vmax, tmax):
    """
    this maps a value between two ranges that both start at 0.

    :param v: value to be scaled
    :param vmax: maximum value of original value range
    :param tmax: maximum value or target range
    :return:
    """
    return (v/vmax)*tmax


RAW_FORCES = 'raw'
BINARY_FORCES = 'binary'

SPARSE_REWARDS = 'sparse'
CONT_REWARDS = 'continuous'


class LoadCellTactileEnv(BulletRobotEnv):

    def __init__(self, joints, force_noise_mu=None, force_noise_sigma=None, force_smoothing=None,
                 target_forces=None, force_threshold=None, force_type=None, reward_type=None,
                 force_sampling_range=None, velocity_rew_coef=None, object_velocity_rew_coef=None,
                 accel_rew_coef=None, *args, **kwargs):

        self.force_smoothing = force_smoothing if force_smoothing is not None else 4
        self.force_noise_mu = force_noise_mu if force_noise_mu is not None else 0.0
        self.force_noise_sigma = force_noise_sigma if force_noise_sigma is not None else 0.0077
        self.force_threshold = force_threshold if force_threshold is not None else 3 * self.force_noise_sigma
        self.force_sampling_range = force_sampling_range

        self.force_buffer_r = deque(maxlen=self.force_smoothing)
        self.force_buffer_l = deque(maxlen=self.force_smoothing)

        self.success_threshold = 5 * self.force_noise_sigma

        self.accel_rew_coef = accel_rew_coef
        self.velocity_rew_coef = velocity_rew_coef
        self.object_velocity_rew_coef = object_velocity_rew_coef

        if target_forces is not None:
            self.target_forces = np.array(target_forces)
        else:
            self.target_forces = np.array([10.0, 10.0])

        self.fmax = np.sum(np.abs(target_forces))

        self.force_type = force_type or RAW_FORCES
        self.reward_type = reward_type or CONT_REWARDS
        assert self.reward_type in {CONT_REWARDS, SPARSE_REWARDS}, f"unknown reward type {self.reward_type}"

        self.current_forces = np.array([0.0, 0.0])
        self.current_forces_raw = np.array([0.0, 0.0])

        self.last_forces = np.array([0.0, 0.0])
        self.last_forces_raw = np.array([0.0, 0.0])

        self.in_contact = np.array([False, False])

        self.force_rew = -100
        self.vel_rew = -100
        self.accel_rew = -100
        self.obj_vel_rew = -100

        self.rew = -100

        if self.force_type not in [RAW_FORCES, BINARY_FORCES]:
            print(f"unknown force type: {self.force_type}")
            exit(-1)

        if self.reward_type not in [SPARSE_REWARDS, CONT_REWARDS]:
            print(f"unknown reward type: {self.reward_type}")
            exit(-1)

        BulletRobotEnv.__init__(self, joints=joints, *args, **kwargs)

        self.vmax = np.abs(list(self.max_joint_velocities.values())[0])

    # BulletRobotEnv methods
    # ----------------------------

    def _transform_forces(self, force):
        return (force / 100) + np.random.normal(self.force_noise_mu, self.force_noise_sigma)

    def _get_obs(self):
        # get joint positions and velocities from superclass
        joint_states = super(LoadCellTactileEnv, self)._get_obs()

        if self.objectId:
            # store last forces
            self.last_forces = self.current_forces.copy()
            self.last_forces_raw = self.current_forces_raw.copy()

            # get current contact forces
            f_r, contact_r = self._get_contact_force(self.robotId, self.objectId,
                                                     self.robot_link_to_index['gripper_right_finger_link'],
                                                     self.object_link_to_index['object_link'])
            self.force_buffer_r.append(f_r)

            f_l, contact_l = self._get_contact_force(self.robotId, self.objectId,
                                                     self.robot_link_to_index['gripper_left_finger_link'],
                                                     self.object_link_to_index['object_link'])
            self.force_buffer_l.append(f_l)

            self.in_contact = np.array([contact_r, contact_l])

            # although forces are called "raw", the are averaged to be as close as possible to the real data.
            self.current_forces_raw = np.array([
                np.mean(self.force_buffer_r),
                np.mean(self.force_buffer_l)
            ])

            # calculate current forces based on force type
            if self.force_type == BINARY_FORCES:
                self.current_forces = (np.array(self.current_forces_raw) > self.force_threshold).astype(np.float32)
            elif self.force_type == RAW_FORCES:
                self.current_forces = self.current_forces_raw.copy()
            else:
                print(f"unknown force type: {self.force_type}")
                exit(-1)

        return np.concatenate([joint_states, force_delta(self.current_forces_raw, self.target_forces)])

    def _is_success(self):
        """If the force delta between target and current force is smaller than the force threshold, it's a success.
        Note, that we use the observation forces here that are averaged over the last k samples. This may lead to
        this function returning False even though the desired force was hit for one sample (see tactile_demo). The
        alternative would be to calculate the success on data that differs from the observation, which an agent would
        not have access too. We assume that that behavior would be more confusing for an agent than it would be helpful.
        """
        delta_f = force_delta(self.current_forces_raw, self.target_forces)
        return np.all((np.abs(delta_f) < self.success_threshold)).astype(np.float32)

    def _compute_reward(self):
        if self.reward_type == CONT_REWARDS:
            # reward for force delta minimization
            delta_f = force_delta(self.current_forces_raw, self.target_forces)
            delta_f_sum = np.sum(np.abs(delta_f))
            self.force_rew = - map_in_range(delta_f_sum, self.fmax, 1.0)

            if self.velocity_rew_coef is not None:

                # self.vel_rew = -1+map_in_range(total_vel, self.total_max_vel, self.velocity_rew_coef)
                if np.any(self.in_contact):
                    total_vel = np.sum(np.abs(self.current_vel))
                    self.vel_rew = -self.velocity_rew_coef*(map_in_range(total_vel, 0.16, 1.0)**2)
                    # self.vel_rew = -1+(1-self.velocity_rew_coef*total_vel)
                else:
                    self.vel_rew = -1
            else:
                self.vel_rew = -1.0

            if self.accel_rew_coef is not None:
                total_acc = np.sum(np.abs(self.current_acc))
                self.accel_rew = -map_in_range(total_acc, 2*self.vmax, self.accel_rew_coef)
            else:
                self.accel_rew = 0.0

            if self.object_velocity_rew_coef is not None:
                obj_v = np.abs(np.linalg.norm(np.sum(self.in_contact)*self.get_object_velocity()[0]))
                self.obj_vel_rew = -map_in_range(obj_v, 0.18, self.object_velocity_rew_coef)
            else:
                self.obj_vel_rew = 0.0

            self.rew = self.force_rew + self.vel_rew + self.obj_vel_rew + self.accel_rew
            return self.rew
        elif self.reward_type == SPARSE_REWARDS:
            is_goal = (np.abs(force_delta(self.current_forces_raw, self.target_forces)) < self.force_threshold).astype(np.int8)

            if self.acceleration_penalty:
                acc_penalty = np.array(np.abs(self.current_acc) > self.acceleration_penalty).astype(np.int8)
            else:
                acc_penalty = np.array([0.0, 0.0])

            if self.force_delta_penalty:
                fd_penalty = np.array(np.abs((self.current_forces_raw - self.last_forces_raw) / self.dt) > self.force_delta_penalty).astype(np.int8)
            else:
                fd_penalty = np.array([0.0, 0.0])

            return np.sum(is_goal)-np.sum(acc_penalty)-np.sum(fd_penalty)

    def _reset_callback(self):
        if self.force_sampling_range:
            assert len(self.force_sampling_range) == 2
            self.target_forces = np.around(np.full((2,), np.random.uniform(*self.force_sampling_range)), 3)
            self.fmax = np.sum(np.abs(self.target_forces))


class GripperTactileEnv(LoadCellTactileEnv):

    def __init__(self, initial_state=None, *args, **kwargs):

        joints = [
            'gripper_right_finger_joint',
            'gripper_left_finger_joint',
        ]

        initial_state = initial_state if initial_state is not None else [
            0.045,
            0.045,
        ]

        max_joint_velocities = {
            'gripper_right_finger_joint': 0.08,
            'gripper_left_finger_joint': 0.08,
        }

        if 'object_pos' not in kwargs:
            kwargs.update({
                'object_pos': [0.04, 0.02, 0.6]
            })
        if 'object_model' not in kwargs:
            kwargs.update({
                'object_model': "objects/object.urdf"
            })

        LoadCellTactileEnv.__init__(self,
                                    joints=joints,
                                    initial_state=initial_state,
                                    max_joint_velocities=max_joint_velocities,
                                    cam_yaw=120.5228271484375,
                                    cam_pitch=-68.42454528808594,
                                    cam_distance=1.1823151111602783,
                                    cam_target_position=(-0.2751278877258301, -0.15310688316822052, -0.27969369292259216),
                                    robot_model="gripper_tactile.urdf",
                                    robot_pos=[0.0, 0.0, 0.27],
                                    *args,
                                    **kwargs)


class TIAGoTactileEnv(LoadCellTactileEnv):

    def __init__(self, initial_state=None, *args, **kwargs):

        joints = [
            'torso_lift_joint',
            'arm_1_joint',
            'arm_2_joint',
            'arm_3_joint',
            'arm_4_joint',
            'arm_5_joint',
            'arm_6_joint',
            'arm_7_joint',
            'gripper_right_finger_joint',
            'gripper_left_finger_joint',
        ]
        
        initial_state = initial_state or [
             0.,
             2.71,
             -0.173,
             1.44,
             1.79,
             0.23,
             -0.0424,
             -0.0209,
             0.045,
             0.045
        ]

        max_joint_velocities = {
            'gripper_right_finger_joint': 0.08,
            'gripper_left_finger_joint': 0.08,
            'torso_lift_joint': 0.07
        }
        
        LoadCellTactileEnv.__init__(self,
                                    joints=joints,
                                    initial_state=initial_state,
                                    max_joint_velocities=max_joint_velocities,
                                    cam_yaw=89.6000747680664,
                                    cam_pitch=-35.40000915527344,
                                    cam_distance=1.6000027656555176,
                                    robot_model="tiago_tactile.urdf",
                                    object_model="objects/object.urdf",
                                    object_pos=[0.73, 0.07, 0.6],
                                    table_model="objects/table.urdf",
                                    table_pos=[0.7, 0, 0.27],
                                    *args, **kwargs)


class GripperTactileCloseEnv(GripperTactileEnv):

    def _compute_reward(self):
        return -np.sum(np.abs(np.array([0.0, 0.0])-self.current_pos[:2]))