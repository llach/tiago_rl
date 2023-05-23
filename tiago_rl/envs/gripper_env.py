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

    def __init__(self, model_path, observation_space, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        MujocoEnv.__init__(
            self,
            model_path=model_path,
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

    def _get_obs(self):
        """ concatenate internal state as observation
        """
        return np.concatenate([
                self.q, 
                self.qdot,
                self.qacc
            ])

    def _get_reward(self):
        raise NotImplementedError
    
    def _is_done(self):
        raise NotImplementedError
    
    def _reset_model(self):
        raise NotImplementedError

    def reset_model(self):
        """ reset data, set joints to initial positions and randomize
        """

        self.model = self._reset_model()
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # load data, set starting joint values (open gripper)
        self.data  = mujoco.MjData(self.model)
        self.data.qpos[self._name_2_qpos_id("gripper_left_finger_joint")]  = 0.045
        self.data.qpos[self._name_2_qpos_id("gripper_right_finger_joint")] = 0.045

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
        
        return (
            self._get_obs(),
            self._get_reward(),
            self._is_done(),  # terminated
            False,  # truncated
            {},     # info
        )
    