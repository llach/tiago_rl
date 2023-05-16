import os
import numpy as np
import mujoco as mj
import mujoco_viewer

import xml.etree.ElementTree as ET

def name2id(model, name, obj=mj.mjtObj.mjOBJ_JOINT): return mj.mj_name2id(model, obj, name)

def set_qpos_from_dict(model, data, qdict):
    for joint_name, q in qdict.items():
        joint_id = name2id(model, joint_name)
        joint_qidx = model.jnt_qposadr[joint_id] # joint_id → qpos index
        data.qpos[joint_qidx] = q

def set_ctrl_from_dict(model, data, cdict):
    for act_name, qdes in cdict.items():
        act_id = name2id(model, act_name, mj.mjtObj.mjOBJ_ACTUATOR)
        data.ctrl[act_id] = qdes

def get_body_pos(model, bname):
    body_id = name2id(model, bname, mj.mjtObj.mjOBJ_BODY)
    return model.body_pos[body_id]

def set_body_pos(model, bname, pos):
    body_id = name2id(model, bname, mj.mjtObj.mjOBJ_BODY)
    model.body_pos[body_id] = pos

np.set_printoptions(suppress=True, precision=3)

from tiago_rl.envs.gripper_env import GripperEnv

env = GripperEnv(render_mode="human")

for _ in range(100):
    env.step([0.03, 0.025])
exit(0)


object_pos = np.array([0,0,0.67])
object_pos[1] = round(np.random.uniform(-0.03, 0.03), 3)

raw = ET.parse("/Users/llach/repos/tiago_mj/force_gripper.xml")
root = raw.getroot()
obj = root.findall(".//body[@name='object']")[0]
obj.attrib['pos'] = ' '.join(map(str, object_pos))

model = mj.MjModel.from_xml_string(ET.tostring(raw.getroot(), encoding='utf8', method='xml'))
data = mj.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data, title="Force Gripper")

# fix camera on TIAGo
viewer.cam.azimuth      = -160
viewer.cam.distance     = 0.8
viewer.cam.elevation    = -45
viewer.cam.lookat       = [0.006, 0.0, 0.518]

# viewer.vopt.frame = mj.mjtFrame.mjFRAME_BODY # mjFRAME_BODY | mjFRAME_WORLD | mjFRAME_CONTACT
# viewer.vopt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1

set_qpos_from_dict(model, data, {
    "gripper_right_finger_joint": 0.045, 
    "gripper_left_finger_joint" : 0.045
})
set_body_pos(model, "object", [0,-0.03, 1.67])
print(get_body_pos(model, "object"))

# simulate and render
for _ in range(1000):

    set_ctrl_from_dict(model, data, {
        "gripper_right_finger_joint": 0.01, 
        "gripper_left_finger_joint" : 0.01
    })

    mj.mj_step(model, data)
    viewer.render()

viewer.close()