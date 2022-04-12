from enum import Enum

class ObsConfig(str, Enum):
    GOAL_DELTA = "goal_delta"
    POSITIONS = "positions"
    VELOCITIES = "velocities"
    FORCES = "forces"

class RewardTypes(str, Enum):
    CONTINUOUS = "continuous"
    SPARSE = "sparse"    

class ControlMode(str, Enum):
    POS_CTRL = 'pos'
    VMAX_CTRL = 'v_max'
    POS_DELTA_CTRL = 'pos_delta'

class ActionPenalty(str, Enum):
    NONE = "none"
    ERROR = "error"
    MAG = "magnitude"
    ERR_DELTA_VEL = "error_delta_velocity"