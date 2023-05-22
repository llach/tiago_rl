import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 0.8,
    "azimuth": -160,
    "elevation": -45,
    "lookat": [0.006, 0.0, 0.518]
}

class GripperEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    INITIAL_OBJECT_POS = np.array([0,0,0.67])

    def __init__(self, ftheta=0.01, fgoal_range=[0.05, 0.6], **kwargs):
        self.ftheta = ftheta    # threshold for contact/no contact
        self.fgoal_range = fgoal_range # sampling range for fgoal

        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            model_path="/Users/llach/repos/tiago_mj/force_gripper.xml",
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # reload the model with environment randomization
        self.reset_model()
    
    def _name_2_qpos_id(self, name):
        """ given a joint name, return their `qpos`-array address
        """
        jid =self.data.joint(name).id
        return self.model.jnt_qposadr[jid]
    
    def _set_action_space(self):
        """ torso joint is ignored, this env is for gripper behavior only
        """
        self.action_space = Box(
            low  = np.array([0.0, 0.0]), 
            high = np.array([0.045, 0.045]), 
            dtype=np.float32
        )
        return self.action_space
    
    def _make_action(self, ain):
        """ creates full `data.ctrl`-compatible array even though some joints are not actuated 
        """
        aout = np.zeros_like(self.data.ctrl)
        aout[self.data.actuator("gripper_left_finger_joint").id]  = ain[0]
        aout[self.data.actuator("gripper_right_finger_joint").id] = ain[1]

        return aout
    
    def _update_state(self):
        """ updates internal state variables that may be used as observations
        """
        ### update relevant robot state variables 
        # joint states
        self.q = np.array([
            self.data.joint("gripper_left_finger_joint").qpos[0],
            self.data.joint("gripper_right_finger_joint").qpos[0]
        ])
        self.qdot = np.array([
            self.data.joint("gripper_left_finger_joint").qvel[0],
            self.data.joint("gripper_right_finger_joint").qvel[0]
        ])
        self.qacc = np.array([
            self.data.joint("gripper_left_finger_joint").qacc[0],
            self.data.joint("gripper_right_finger_joint").qacc[0]
        ])

        # forces
        self.forces = np.array([
            self.data.sensor("left_touch_sensor").data[0],
            self.data.sensor("right_touch_sensor").data[0],
        ])
        self.in_contact = self.forces > self.ftheta

        # object state
        self.objv = np.linalg.norm(self.data.joint("object_joint").qvel[:3])
        self.objw = np.linalg.norm(self.data.joint("object_joint").qvel[3:])

    def _get_obs(self):
        """ concatenate internal state as observation
        """
        return np.concatenate([
                self.q, 
                self.qdot, 
                self.forces, 
                self.in_contact, 
                [self.objv, self.objw]
            ])

    def reset_model(self):
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

        # create model from modified XML and set vis parameters
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xmlmodel.getroot(), encoding='utf8', method='xml'))
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # load data, set starting joint values (open gripper)
        self.data  = mujoco.MjData(self.model)
        self.data.qpos[self._name_2_qpos_id("gripper_left_finger_joint")]  = 0.045
        self.data.qpos[self._name_2_qpos_id("gripper_right_finger_joint")] = 0.045

        # sample goal force
        self.fgoal = round(np.random.uniform(*[0.05, 0.6]), 3)

        # update renderer's pointers, otherwise scene will be empty
        self.mujoco_renderer.model = self.model
        self.mujoco_renderer.data  = self.data
        
        # viewers' models also need to be updated 
        if len(self.mujoco_renderer._viewers)>0:
            for _, v in self.mujoco_renderer._viewers.items():
                v.model = self.model
                v.data = self.data

        self._update_state()
        return self._get_obs()

    def step(self, a):
        """
        action: [q_left, q_right]

        returns:
            observations
            reward
            terminated
            truncated
            info
        """
        
        # `self.do_simulation` invovled an action space shape check that this environment won't pass due to underactuation
        self._step_mujoco_simulation(self._make_action(a), self.frame_skip)
        if self.render_mode == "human":
            self.render()

        # update internal state variables
        self._update_state()

        # calculate reward
        fdelta = self.fgoal - np.sum(self.forces)
        fdelta = -np.abs(fdelta) # abs: same sign for overshooting forces and too little force; - penalize differences from fgoal
        
        return (
            self._get_obs(),
            fdelta,
            False,  # terminated
            False,  # truncated
            {},     # info
        )