import mujoco
import numpy as np

from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

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

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64) # TODO check

        MujocoEnv.__init__(
            self,
            model_path="/Users/llach/repos/tiago_mj/force_gripper.xml",
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self._reset_simulation()
    
    def _name_2_qpos_id(self, name):
        """ given a joint name, return their `qpos`-array address
        """
        jid =self.data.joint(name).id
        return self.model.jnt_qposadr[jid]
    
    def _reset_simulation(self):
        """ reset data, set joints to initial positions and randomize
        """
        mujoco.mj_resetData(self.model, self.data)
        
        self.data.qpos[self._name_2_qpos_id("gripper_left_finger_joint")]  = 0.045
        self.data.qpos[self._name_2_qpos_id("gripper_right_finger_joint")] = 0.045

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

    def step(self, a):
        """
        action: [q_left, q_right]
        """
        
        # `self.do_simulation` invovled an action space shape check that this environment won't pass due to underactuation
        self._step_mujoco_simulation(self._make_action(a), self.frame_skip)
        if self.render_mode == "human":
            self.render()

        return (
            [],
            [],
            False,
            False,
            {},
        )