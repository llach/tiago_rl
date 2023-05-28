import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium.spaces import Box

from tiago_rl import safe_rescale, total_contact_force
from tiago_rl.envs import GripperEnv


class GripperTactileEnv(GripperEnv):

    INITIAL_OBJECT_POS = np.array([0,0,0.67])

    def __init__(self, ftheta=0.08, fgoal_range=[0.05, 0.5], fmax=0.6, obj_pos_range=[-0.03, 0.03], **kwargs):
        self.fmax = fmax                # maximum force
        self.ftheta = ftheta            # threshold for contact/no contact
        self.fgoal_range = fgoal_range  # sampling range for fgoal
        self.obj_pos_range = obj_pos_range

        observation_space = Box(low=-1, high=1, shape=(8,), dtype=np.float64)

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
            np.sum(np.abs(total_contact_force(self.model, self.data, "object", "left_finger_bb")[0])),
            0
            # total_contact_force(self.model, self.data, "object", "right_finger_bb")[0][2]
            # self.data.sensor("left_touch_sensor").data[0],
            # self.data.sensor("right_touch_sensor").data[0],
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
                safe_rescale(self.forces, [0, self.fmax]), 
                safe_rescale(self.force_deltas, [-self.fgoal, self.fgoal]),
            ])

    def _get_reward(self):
        fdelta = np.abs(self.fgoal - self.forces)
        fdelta = np.clip(fdelta, 0, self.fgoal)

        rforce = np.sum(1-(fdelta/self.fgoal))

        return rforce #- self._qdot_penalty()
    
    def _is_done(self): return False

    def _reset_model(self):
        """ reset data, set joints to initial positions and randomize
        """
        xmlmodel = ET.parse(self.fullpath)
        root = xmlmodel.getroot()

        #-----------------------------
        # random object start 

        object_pos    = self.INITIAL_OBJECT_POS.copy()
        object_pos[1] = round(np.random.uniform(*self.obj_pos_range), 3)

        obj = root.findall(".//body[@name='object']")[0]
        obj.attrib['pos'] = ' '.join(map(str, object_pos))

        # sample goal force
        self.fgoal = round(np.random.uniform(*self.fgoal_range), 3)

        # create model from modified XML
        return mujoco.MjModel.from_xml_string(ET.tostring(xmlmodel.getroot(), encoding='utf8', method='xml'))

    def set_goal(self, x): self.fgoal = x