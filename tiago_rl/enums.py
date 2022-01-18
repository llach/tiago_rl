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
    VEL_CTRL = 'vel'
    POS_DELTA_CTRL = 'pos_delta'
