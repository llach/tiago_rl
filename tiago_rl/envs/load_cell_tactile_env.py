import numpy as np
import pybullet as p

from tiago_rl.envs import BulletRobotEnv
from tiago_rl.envs.utils import link_to_idx


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

    def __init__(self, joints, force_noise_mu=None, force_noise_sigma=None, target_force=None, force_type=None, reward_type=None, object_velocity_rew_coef=None, width_range=None, location_sampling=False, shape_sampling=False, *args, **kwargs):

        self.force_noise_mu = force_noise_mu if force_noise_mu is not None else 0.0
        self.force_noise_sigma = force_noise_sigma if force_noise_sigma is not None else 0.0077
        
        self.force_threshold =  3 * self.force_noise_sigma
        self.success_threshold = 5 * self.force_noise_sigma

        self.object_velocity_rew_coef = object_velocity_rew_coef

        self.target_force = target_force
        if type(self.target_force) == float:
            self.target_forces = np.array(2*[self.target_force])
        else:
            self.target_forces = np.array([0.0, 0.0])

        self.fmax = np.sum(np.abs(self.target_forces))

        self.force_type = force_type or RAW_FORCES
        self.reward_type = reward_type or CONT_REWARDS
        assert self.reward_type in {CONT_REWARDS, SPARSE_REWARDS}, f"unknown reward type {self.reward_type}"

        self.current_forces = np.array([0.0, 0.0])
        self.current_forces_raw = np.array([0.0, 0.0])

        self.last_forces = np.array([0.0, 0.0])
        self.last_forces_raw = np.array([0.0, 0.0])

        self.in_contact = np.array([False, False])

        self.force_rew = -100
        self.obj_vel_rew = -100

        self.rew = -100

        if self.force_type not in [RAW_FORCES, BINARY_FORCES]:
            print(f"unknown force type: {self.force_type}")
            exit(-1)

        if self.reward_type not in [SPARSE_REWARDS, CONT_REWARDS]:
            print(f"unknown reward type: {self.reward_type}")
            exit(-1)

        # Environment Variation Variables
        self.width_range = width_range
        self.location_sampling = location_sampling
        self.shape_sampling = shape_sampling

        self.olx = 0.04
        self.oly = 0.0
        self.olz = 0.588

        self.r = 0.02

        self.obj_col_id = None
        self.object_type = None
        self.object_id = None

        BulletRobotEnv.__init__(self, joints=joints, *args, **kwargs)
        
    # BulletRobotEnv methods
    # ----------------------------

    def _transform_forces(self, force):
        return (force / 100) + np.random.normal(self.force_noise_mu, self.force_noise_sigma)

    def _get_obs(self):
        # get joint positions and velocities from superclass
        joint_states = super(LoadCellTactileEnv, self)._get_obs()

        if self.object_id:
            # store last forces
            self.last_forces = self.current_forces.copy()
            self.last_forces_raw = self.current_forces_raw.copy()

            # get current contact forces
            f_r, contact_r = self._get_contact_force(self.robotId, self.object_id,
                                                     self.robot_link_to_index['gripper_right_finger_link'],
                                                     self.object_link_to_index['link0'])

            f_l, contact_l = self._get_contact_force(self.robotId, self.object_id,
                                                     self.robot_link_to_index['gripper_left_finger_link'],
                                                     self.object_link_to_index['link0'])

            self.in_contact = np.array([contact_r, contact_l])

            # although forces are called "raw", the are averaged to be as close as possible to the real data.
            self.current_forces_raw = np.array([f_r, f_l])

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

            if self.object_velocity_rew_coef is not None:
                obj_v = np.abs(np.linalg.norm(np.sum(self.in_contact)*self.get_object_velocity()[0]))
                self.obj_vel_rew = -map_in_range(obj_v, 0.18, self.object_velocity_rew_coef)
            else:
                self.obj_vel_rew = 0.0

            self.rew = self.force_rew + self.obj_vel_rew
            return self.rew
        elif self.reward_type == SPARSE_REWARDS:
            is_goal = (np.abs(force_delta(self.current_forces_raw, self.target_forces)) < self.force_threshold).astype(np.int8)
            return np.sum(is_goal)

    def _reset_callback(self):
        # target force sampling
        if type(self.target_force) == list:
            assert len(self.target_force) == 2
            self.target_forces = np.around(np.full((2,), np.random.uniform(*self.target_force)), 3)
        else:
            self.target_forces = np.array(2*[self.target_force])
        self.fmax = np.sum(np.abs(self.target_forces))

        # object width variation
        if self.width_range is None:
            self.r = 0.02
        else:
            self.r = np.round(np.random.uniform(self.width_range[0], self.width_range[1]), 4)

        if self.shape_sampling:
            self.object_type = np.random.choice([p.GEOM_CYLINDER, p.GEOM_BOX])
        else:
            self.object_type = p.GEOM_CYLINDER

        # create collision and visual objects
        height = 0.1
        he = [0.02, self.r, height/2]
        self.obj_col_id = p.createCollisionShape(self.object_type, halfExtents=he, height=height, radius=self.r)
        self.obj_vis_id = p.createVisualShape(self.object_type, halfExtents=he, length=height, radius=self.r, rgbaColor=list(np.random.uniform(0,1,[3])) + [1])

        # sample object location
        if self.location_sampling:
            l = 2*0.045*0.95 # we only use 95% of the opening's width
            f = l-2*self.r
            self.oly = np.round(np.random.uniform(-f/2, f/2), 4)
        else:
            self.oly = self.oly

        # create body and apply stiffness parameters
        self.object_id = p.createMultiBody(2.0, self.obj_col_id, self.obj_vis_id, [self.olx, self.oly, self.olz], [0, 0, 0, 1])
        p.changeDynamics(self.object_id, -1, lateralFriction=1.0, rollingFriction=1.0, contactStiffness=10000, contactDamping=100)

        # finally, create link mapping
        self.object_link_to_index = link_to_idx(self.object_id)
    
    def get_object_velocity(self):
        if self.object_id is not None:
            return p.getBaseVelocity(self.object_id)
        else:
            return [0.0, 0.0]


