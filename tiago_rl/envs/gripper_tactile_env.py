import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium.spaces import Box

from tiago_rl import safe_rescale, total_contact_force
from tiago_rl.envs import GripperEnv


class GripperTactileEnv(GripperEnv):

    QY_SGN_l =  1
    QY_SGN_r = -1
    SOLREF = [0.02, 1]
    SOLIMP = [0, 0.95, 0.01, 0.2, 2] # dmin is set to 0 to allow soft contacts
    INITIAL_OBJECT_POS = np.array([0,0,0.67])

    def __init__(self, ftheta=0.05, fgoal_range=[0.4, 0.6], fmax=0.75, obj_pos_range=[0, 0], beta=0.6, gamma=1.0, **kwargs):
        self.fmax = fmax                # maximum force
        self.ftheta = ftheta            # threshold for contact/no contact
        self.fgoal_range = fgoal_range  # sampling range for fgoal
        self.gamma = gamma
        self.obj_pos_range = obj_pos_range

        observation_space = Box(low=-1, high=1, shape=(10,), dtype=np.float64)

        # solver parameters that control object deformation and contact force behavior
        self.solref = self.SOLREF
        self.solimp = self.SOLIMP

        GripperEnv.__init__(
            self,
            model_path="/Users/llach/repos/tiago_mj/force_gripper.xml",
            observation_space=observation_space,
            qinit_range=[0.03, 0.03],
            beta=beta,
            **kwargs,
        )

    def _update_state(self):
        """ updates internal state variables that may be used as observations
        """
        super()._update_state()

        # forces
        self.forces = np.array([
            np.sum(np.abs(total_contact_force(self.model, self.data, "object", "left_finger_bb")[0])),
            np.sum(np.abs(total_contact_force(self.model, self.data, "object", "right_finger_bb")[0]))
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
                self.in_contact,
            ])

    def _object_proximity_reward(self):
        """ fingers don't move towards the object sometimes → encourage them with small, positive rewards
        """
        # clipping the deltas at 0 will allow π to penetrate the object without penalty
        dl = 1-np.clip(self.q[0]-self.qo_l, 0, self.doq_l)/self.doq_l
        dr = 1-np.clip(self.q[0]-self.qo_r, 0, self.doq_r)/self.doq_r
        return self.gamma*(dl+dr)

    def _get_reward(self):
        fdelta = np.abs(self.fgoal - self.forces)
        fdelta = np.clip(fdelta, 0, self.fgoal)

        rforce = np.sum((1-(fdelta/self.fgoal)))

        return rforce + self._object_proximity_reward() + np.sum(self.in_contact) - self._qdot_penalty()
        # return rforce + self._object_proximity_reward() - self._qdot_penalty()
    
    def _is_done(self): return False

    def _reset_model(self):
        """ reset data, set joints to initial positions and randomize
        """
        xmlmodel = ET.parse(self.fullpath)
        root = xmlmodel.getroot()

        #-----------------------------
        # random object start 

        self.oy = round(np.random.uniform(*self.obj_pos_range), 3) # object y position
        object_pos    = self.INITIAL_OBJECT_POS.copy()
        object_pos[1] = self.oy

        obj = root.findall(".//body[@name='object']")[0]
        obj.attrib['pos']    = ' '.join(map(str, object_pos))

        objgeom = obj.findall(".//geom")[0]
        objgeom.attrib['solimp'] = ' '.join(map(str, self.solimp))
        
        # store object half-width (radius for cylinders)
        self.ow = float(objgeom.attrib['size'].split(' ')[0])
        assert np.abs(self.ow) > np.abs(self.oy), "|ow| > |oy|"

        # sample goal force
        self.fgoal = round(np.random.uniform(*self.fgoal_range), 3)

        # signs for object q calculation
        sgnl = np.sign(self.oy)*self.QY_SGN_l
        sgnr = np.sign(self.oy)*self.QY_SGN_r
        # if oy is zero, sign(oy) also is, then it's fine to not do the assertion
        if self.oy != 0: assert sgnl != sgnr, "sgnl != sgnr"

        self.qo_l = sgnl*self.oy + self.ow 
        self.qo_r = sgnr*self.oy + self.ow

        # distance between object and finger. TODO do we need to account for finger width?
        self.doq_l = self.qinit_l-self.qo_l 
        self.doq_r = self.qinit_r-self.qo_r

        # create model from modified XML
        return mujoco.MjModel.from_xml_string(ET.tostring(xmlmodel.getroot(), encoding='utf8', method='xml'))

    def set_goal(self, x): self.fgoal = x
    def set_solver_parameters(self, solimp=None, solref=None):
        """ see https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters
        """
        self.solimp = solimp
        self.solref = solref