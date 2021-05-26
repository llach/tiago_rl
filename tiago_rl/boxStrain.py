import time
import pybullet_data

import numpy as np
import pybullet as p

# configure simulator
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # allows loading of pybullet_data URDFs
p.setGravity(0, 0, -10)

# set camera to view at scene
p.resetDebugVisualizerCamera(1.1823151111602783, 120.5228271484375, -68.42454528808594, (-0.2751278877258301, -0.15310688316822052, -0.27969369292259216))

# disable rendering of world axis and GUI elements
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# load objects
planeId = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.01])
cylId = p.loadURDF("./assets/objects/cylinder.urdf", basePosition=[0.04, 0.03, 0.6])

# load robot
robId = p.loadURDF("assets/boxBotStrain.urdf", basePosition=[0.0, 0.0, 0.27])

# create jointName to jointIndex mapping
name2Idx = {key.decode(): value for (value, key) in [p.getJointInfo(robId, i)[:2] for i in range(p.getNumJoints(robId))]}

# set initial grasping position
initialPositions = [
    ['gripper_right_finger_joint', 0.045],
    ['gripper_left_finger_joint', 0.045],
    ['torso_to_arm', 0.00]
]

for jn, q in initialPositions:
    p.resetJointState(robId, name2Idx[jn], q)


numSteps = 500
gripper_qs = np.linspace(0.045, 0.025, num=numSteps)
torso_qs = np.linspace(0, 0.05, num=numSteps)

# wait a bit for things to settle in simulation
for i in range(100):
    p.stepSimulation()
    time.sleep(1./250.)

# step simulation
for i in range(10000):
    p.stepSimulation()
    time.sleep(1./250.)

    if i < numSteps:
        p.resetJointState(robId, name2Idx['gripper_right_finger_joint'], gripper_qs[i])
        p.resetJointState(robId, name2Idx['gripper_left_finger_joint'], gripper_qs[i])
        # p.resetJointState(robId, name2Idx['torso_to_arm'], torso_qs[i])

    # c = p.getDebugVisualizerCamera()
    # print(f"{c[-2]}, {c[-4]}, {c[-3]}, {c[-1]}")
p.disconnect()
