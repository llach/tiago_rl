import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium.spaces import Box

from tiago_rl import safe_rescale
from tiago_rl.envs import GripperEnv


class GripperTactileEnv(GripperEnv):

    QY_SGN_l =  1
    QY_SGN_r = -1
    SOLREF = [0.02, 1]
    SOLIMP = [0, 0.95, 0.01, 0.2, 2] # dmin is set to 0 to allow soft contacts
    INITIAL_OBJECT_POS = np.array([0,0,0.67])

    def __init__(self, fgoal_range=[0.5, 0.6], obj_pos_range=[0, 0], alpha=1.0, beta=1.0, gamma=1.0, **kwargs):
        self.fgoal_range = fgoal_range  # sampling range for fgoal
        self.alpha = alpha
        self.gamma = gamma
        self.obj_pos_range = obj_pos_range

        observation_space = Box(low=-1, high=1, shape=(12,), dtype=np.float64)

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

        # force deltas to goal
        self.force_deltas = self.fgoal - self.forces

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
        dr = 1-np.clip(self.q[1]-self.qo_r, 0, self.doq_r)/self.doq_r
        return self.gamma*(dl+dr)
    
    def _force_reward(self):
        deltaf = self.fgoal - self.forces
        
        rforce = 0
        for df in deltaf:
            if df < -self.ftheta: # overshooting
                rforcei = 1-10*(np.abs(df)/self.frange_upper)
            elif df > self.ftheta:
                rforcei = 1-(df/self.frange_lower)
            else: rforcei = 1.1 # within goal-band 
            rforce += rforcei
        return self.alpha*rforce

    def _get_reward(self):
        self.r_force   = self._force_reward()
        self.r_obj_prx = self._object_proximity_reward()
        self.r_qdot    = self._qdot_penalty()
        self.r_qacc    = self._qacc_penalty()

        return self.r_force + self.r_obj_prx + np.sum(self.in_contact) - self.r_qdot - self.r_qacc
    
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
        self.set_goal(round(np.random.uniform(*self.fgoal_range), 3))

        # signs for object q calculation
        sgnl = np.sign(self.oy)*self.QY_SGN_l
        sgnr = np.sign(self.oy)*self.QY_SGN_r
        # if oy is zero, sign(oy) also is, then it's fine to not do the assertion
        if self.oy != 0: assert sgnl != sgnr, "sgnl != sgnr"

        self.qo_l = sgnl*self.oy + self.ow 
        self.qo_r = sgnr*self.oy + self.ow

        # distance between object and finger (doesn't take finger width into account)
        self.doq_l = self.qinit_l-self.qo_l 
        self.doq_r = self.qinit_r-self.qo_r

        # create model from modified XML
        return mujoco.MjModel.from_xml_string(ET.tostring(xmlmodel.getroot(), encoding='utf8', method='xml'))

    def set_goal(self, x): 
        # set goal force and calculate interval sizes above and below goal force
        self.fgoal = x
        self.frange_upper = self.fmax - self.fgoal
        self.frange_lower = self.fgoal # fmin is 0

    def set_solver_parameters(self, solimp=None, solref=None):
        """ see https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters
        """
        self.solimp = solimp
        self.solref = solref