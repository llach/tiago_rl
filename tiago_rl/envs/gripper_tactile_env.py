import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium.spaces import Box
from .gripper_env import GripperEnv


class GripperTactileEnv(GripperEnv):

    INITIAL_OBJECT_POS = np.array([0,0,0.67])

    def __init__(self, ftheta=0.08, fgoal_range=[0.05, 0.6], **kwargs):
        self.ftheta = ftheta    # threshold for contact/no contact
        self.fgoal_range = fgoal_range # sampling range for fgoal

        observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64)

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

        # forces
        self.forces = np.array([
            self.data.sensor("left_touch_sensor").data[0],
            self.data.sensor("right_touch_sensor").data[0],
        ])
        self.force_deltas = self.fgoal - self.forces
        self.in_contact = self.forces > self.ftheta

        # object state
        self.objv = np.linalg.norm(self.data.joint("object_joint").qvel[:3])
        self.objw = np.linalg.norm(self.data.joint("object_joint").qvel[3:])

    def _get_obs(self):
        """ concatenate internal state as observation
        """ 
        return np.concatenate([
                super()._get_obs(),
                self.forces, 
                self.force_deltas,
                self.in_contact, 
                [self.objv, self.objw]
            ])

    def _get_reward(self):
        fdelta = np.abs(self.fgoal - np.sum(self.forces))
    
        if fdelta>self.ftheta: return 0 
        return np.e**(5*(((self.ftheta-fdelta)/self.ftheta)-1))
    
    def _is_done(self): return False

    def _reset_model(self):
        """ reset data, set joints to initial positions and randomize
        """
        xmlmodel = ET.parse(self.fullpath)
        root = xmlmodel.getroot()

        #-----------------------------
        # random object start 

        object_pos = self.INITIAL_OBJECT_POS.copy()
        object_pos[1] = round(np.random.uniform(-0.03, 0.03), 3)

        obj = root.findall(".//body[@name='object']")[0]
        obj.attrib['pos'] = ' '.join(map(str, object_pos))

        # sample goal force
        self.fgoal = round(np.random.uniform(*[0.05, 0.6]), 3)

        # create model from modified XML
        return mujoco.MjModel.from_xml_string(ET.tostring(xmlmodel.getroot(), encoding='utf8', method='xml'))
