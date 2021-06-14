import time
import pybullet_data

import numpy as np
import pybullet as p

from collections import deque

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import platform


def getLinkToIdxDict(bodyId):
    d = {p.getBodyInfo(bodyId)[0].decode('UTF-8'): -1, }

    for _id in range(p.getNumJoints(bodyId)):
        _name = p.getJointInfo(bodyId, _id)[12].decode('UTF-8')
        d[_name] = _id

    return d


def calculateForces(contacts):
    if not contacts:
        return 0.0

    # we might want to do impose additional checks upon contacts
    f = [c[9] for c in contacts]
    return np.sum(f)


# app = QtGui.QApplication([])
#
# win = pg.GraphicsLayoutWidget(show=True,)
# win.resize(1000, 600)
# win.setWindowTitle('Force Visualisation')
#
# # Enable antialiasing for prettier plots
# pg.setConfigOptions(antialias=True)
#
# p1 = win.addPlot(title="TA11 Scalar Contact Forces")
#
# curve = p1.plot(pen='y')
# curve2 = p1.plot(pen='r')
forces_l = []
forces_r = []
# ptr = 0
# def updatePlot():
#     global curve, curve2, forces_l, forces_r, ptr, p1
#     curve.setData(forces_l)
#     curve2.setData(forces_r)
#     ptr += 1
#     if platform.system() == 'Linux':
#         app.processEvents()

# configure simulator
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # allows loading of pybullet_data URDFs
p.setGravity(0, 0, -10)
# p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

# set camera to view at scene
p.resetDebugVisualizerCamera(1.1823151111602783, 120.5228271484375, -68.42454528808594, (-0.2751278877258301, -0.15310688316822052, -0.27969369292259216))

# disable rendering of world axis and GUI elements
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# load objects
planeId = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.01])
cylId = p.loadSoftBody("./assets/objects/cylinder.obj", basePosition=[0.04, 0.02, 0.65], baseOrientation=p.getQuaternionFromEuler([1.57,0,0]),
                       scale=1.0,
                       mass=40.0,
                       collisionMargin=0.003,
                       useNeoHookean=0,
                       useBendingSprings=1,
                       useMassSpring=1,
                       springElasticStiffness=40,
                       springDampingStiffness=15,
                       springDampingAllDirections=1,
                       useSelfCollision=1,
                       frictionCoeff=1.0,
                       useFaceContact=1
                       )

#
# visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
#                                     fileName="./assets/objects/cylinder.obj",
#                                     rgbaColor=[1, 0, 1, 1],
#                                     specularColor=[0.4, .4, 0],
#                                     visualFramePosition=[0,0,0],
#                                     meshScale=1)
# collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
#                                           fileName="./assets/objects/cylinder.obj",
#                                           collisionFramePosition=[0,0,0],
#                                           meshScale=1)
#
# p.createMultiBody(baseMass=10,
#                   baseInertialFramePosition=[0, 0, 0],
#                   baseCollisionShapeIndex=collisionShapeId,
#                   baseVisualShapeIndex=cylId,
#                   basePosition=[0.04, 0.02, 0.65],
#                     baseOrientation=p.getQuaternionFromEuler([1.57,0,0]),
#                     useMaximalCoordinates=True)
# stoneId = p.createCollisionShape(p.GEOM_MESH,fileName="./assets/objects/cylinder.obj")
# cylId = p.loadSoftBody("toys/cylinder.obj", basePosition=[0.04, 0.02, 0.8], baseOrientation=p.getQuaternionFromEuler([1.57,0,0]))

# load robot
robId = p.loadURDF("assets/boxBotStrain.urdf", basePosition=[0.0, 0.0, 0.27])

# create jointName to jointIndex mapping
name2Idx = {key.decode(): value for (value, key) in [p.getJointInfo(robId, i)[:2] for i in range(p.getNumJoints(robId))]}

robLink2Idx = getLinkToIdxDict(robId)
# objLink2Idx = getLinkToIdxDict(cylId)

# p.changeDynamics(bodyUniqueId=cylId, linkIndex=-1, contactStiffness=5.5, contactDamping=5.5)


# set initial grasping position
initialPositions = [
    ['gripper_right_finger_joint', 0.045],
    ['gripper_left_finger_joint', 0.045],
    ['torso_to_arm', 0.00]
]

for jn, q in initialPositions:
    p.resetJointState(robId, name2Idx[jn], q)

numSteps = 70
gripper_qs = np.linspace(0.045, 0.01, num=numSteps)
torso_qs = np.linspace(0, 0.05, num=numSteps)

me = 4
fl = deque(maxlen=me)
fr = deque(maxlen=me)


waitSteps = 140
# step simulation
for i in range(140*2):
    p.stepSimulation()
    time.sleep(1./140.)

    # contact_l = p.getContactPoints(bodyA=robId,
    #                                bodyB=cylId,
    #                                linkIndexA=robLink2Idx['gripper_left_finger'],c
    #                                linkIndexB=objLink2Idx['cylinderLink'])
    # fl.append(calculateForces(contact_l))
    #
    # contact_r = p.getContactPoints(bodyA=robId,
    #                                bodyB=cylId,
    #                                linkIndexA=robLink2Idx['gripper_right_finger'],
    #                                linkIndexB=objLink2Idx['cylinderLink'])
    # fr.append(calculateForces(contact_r))
    #
    # forces_l.append(np.mean(fl) / 100 + np.random.normal(0, 0.0077))
    # forces_r.append(np.mean(fr) / 100 + np.random.normal(0, 0.0077))

    if waitSteps < i < waitSteps+numSteps:
        n = i-waitSteps
        p.setJointMotorControl2(bodyUniqueId=robId,
                                jointIndex=name2Idx['gripper_right_finger_joint'],
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_qs[n])

        p.setJointMotorControl2(bodyUniqueId=robId,
                                jointIndex=name2Idx['gripper_left_finger_joint'],
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_qs[n])

    # updatePlot()

    # c = p.getDebugVisualizerCamera()
    # print(f"{c[-2]}, {c[-4]}, {c[-3]}, {c[-1]}")

# use this to store force values to pickle
# import os
# import pickle
#
# with open('{}/simClose.pkl'.format(os.environ['HOME']), 'wb') as f:
#     pickle.dump({
#         'forces_l': forces_l,
#         'forces_r': forces_r
#     }, f, protocol=2)

p.disconnect()
