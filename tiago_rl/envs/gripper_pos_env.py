import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium.spaces import Box
from .gripper_env import GripperEnv
from tiago_rl import safe_rescale


class GripperPosEnv(GripperEnv):

    def __init__(self, max_steps=50, vmax=0.2, valpha=6, eps=None, **kwargs):
        self.vmax = vmax
        self.valpha = valpha
        self.max_steps = max_steps

        self.eps            = 0.0005
        self.in_band        = 0 
        self.qgoal_range    = [0.0, 0.045]

        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

        GripperEnv.__init__(
            self,
            model_path="/Users/llach/repos/tiago_mj/force_gripper.xml",
            observation_space=observation_space,
            **kwargs,
        )

    def _update_state(self):
        """ updates internal state variables that may be used as observations
        """
        super()._update_state()
        self.q_deltas = self.qgoal - self.q

    def _get_obs(self):
        """ concatenate internal state as observation
        """ 
        return np.concatenate([
            safe_rescale(self.q, [0.0, 0.045], [-1,1]),
            safe_rescale(self.q_deltas, [-0.045, 0.045], [-1,1]),
            safe_rescale(self.qdot, [-self.vmax, self.vmax], [-1,1]),
        ])
    
    # # variant I) continuous reward - no restrictions
    # def _get_reward(self):
    #     return -np.sum(np.abs(self.qgoal - self.q))
    
    # # variant II.1) continuous reward inside ε-environment 
    # # DON'T USE, THIS ONE HAS A LOGIC ERROR
    # def _get_reward(self):
    #     deltaq = np.abs(self.qgoal - self.q)
    #     if not np.all(deltaq<self.eps): return 0

    #     # reward is super small since ε is, make it bigger with 1e5
    #     return np.sum(np.abs(self.qgoal - self.q))*1e5

    # # variant II.2) continuous reward inside ε-environment, normalized
    # def _get_reward(self):
        # deltaq = np.abs(self.qgoal - self.q)
        # if not np.all(deltaq<self.eps): return 0

        # return np.sum(1-(deltaq/self.eps))

    # variant III) continuous reward normalized inside ε-environment & velocity penalty
    def _get_reward(self):
        vnorm    = np.clip(np.abs(self.qdot), 0, self.vmax)/self.vmax
        vpenalty = np.e**(self.valpha*((vnorm-1)))
        vpenalty = np.sum(vpenalty)

        deltaq = np.abs(self.qgoal - self.q)
        if not np.all(deltaq<self.eps): return -0.1*vpenalty

        posreward = np.sum(1-(deltaq/self.eps))

        return posreward - 2*vpenalty
    
    def _is_done(self): return False

    def _reset_model(self):
        """ reset data, set joints to initial positions and randomize
        """
        xmlmodel = ET.parse(self.fullpath)
        root = xmlmodel.getroot()

        # random object start 
        wb = root.findall(".//worldbody")[0]
        obj = root.findall(".//body[@name='object']")[0]
        wb.remove(obj)

        # sample goal position
        self.qgoal = round(np.random.uniform(*self.qgoal_range), 4)
        self.in_band = 0

        # create model from modified XML
        return mujoco.MjModel.from_xml_string(ET.tostring(xmlmodel.getroot(), encoding='utf8', method='xml'))
    
    def set_goal(self, g): self.qgoal = g

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # if both q deltas are smaller then the deadband, in_band counter is raised by one
        qdelta = np.abs(self.qgoal - self.q)
        if np.all(qdelta<self.eps) or self.in_band>0:
            self.in_band += 1

        terminated = terminated or self.in_band >= self.max_steps
        return obs, reward, terminated, truncated, info

