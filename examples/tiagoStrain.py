import pybullet as p
import time
import pybullet_data

# configure simulator
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0, 0, -10)

# set camera to view at scene
p.resetDebugVisualizerCamera(1.6000027656555176, 89.6000747680664, -35.40000915527344, (0.0, 0.0, 0.0))
p.configureDebugVisualizer(lightPosition=(30.0, 0.0, 20.0))

# disable rendering of world axis and GUI elements
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# load objects
planeId = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.01])
boxId = p.loadURDF("./assets/objects/table.urdf", basePosition=[0.7, 0, 0.27])
cylId = p.loadURDF("./assets/objects/object.urdf", basePosition=[0.73, 0.07, 0.6])

# load robot
robId = p.loadURDF("./assets/tiago_tactile.urdf")

# create jointName to jointIndex mapping
name2Idx = {key.decode(): value for (value, key) in [p.getJointInfo(robId, i)[:2] for i in range(p.getNumJoints(robId))]}

# set initial grasping position
initialPositions = [
    ['torso_lift_joint', 0.],
    ['head_2_joint', -0.7],
    ['arm_1_joint', 2.71],
    ['arm_2_joint', -0.173],
    ['arm_3_joint', 1.44],
    ['arm_4_joint', 1.79],
    ['arm_5_joint', 0.23],
    ['arm_6_joint', -0.0424],
    ['arm_7_joint', -0.0209],
    ['gripper_right_finger_joint', 0.045],
    ['gripper_left_finger_joint', 0.045]
]

for jn, q in initialPositions:
    p.resetJointState(robId, name2Idx[jn], q)

# step simulation
for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect()
