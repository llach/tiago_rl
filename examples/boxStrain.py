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


app = QtGui.QApplication([])

win = pg.GraphicsLayoutWidget(show=True,)
win.resize(1000, 600)
win.setWindowTitle('Force Visualisation')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot(title="TA11 Scalar Contact Forces")

curve = p1.plot(pen='y')
curve2 = p1.plot(pen='r')
forces_l = []
forces_r = []
ptr = 0
def updatePlot():
    global curve, curve2, forces_l, forces_r, ptr, p1
    curve.setData(forces_l)
    curve2.setData(forces_r)
    ptr += 1
    if platform.system() == 'Linux':
        app.processEvents()

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
cylId = p.loadURDF("../tiago_rl/assets/objects/object.urdf", basePosition=[0.04, 0.02, 0.6])

# load robot
robId = p.loadURDF("assets/gripper_tactile.urdf", basePosition=[0.0, 0.0, 0.27])

# create jointName to jointIndex mapping
name2Idx = {key.decode(): value for (value, key) in [p.getJointInfo(robId, i)[:2] for i in range(p.getNumJoints(robId))]}

robLink2Idx = getLinkToIdxDict(robId)
objLink2Idx = getLinkToIdxDict(cylId)

# set initial grasping position
initialPositions = [
    ['gripper_right_finger_joint', 0.045],
    ['gripper_left_finger_joint', 0.045],
    ['torso_to_arm', 0.00]
]

for jn, q in initialPositions:
    p.resetJointState(robId, name2Idx[jn], q)

numSteps = 70
gripper_qs = np.linspace(0.045, 0.00, num=numSteps)
torso_qs = np.linspace(0, 0.05, num=numSteps)

me = 4
fl = deque(maxlen=me)
fr = deque(maxlen=me)

waitSteps = 100
# step simulation
for i in range(400):
    p.stepSimulation()
    time.sleep(1./140.)

    contact_l = p.getContactPoints(bodyA=robId,
                                   bodyB=cylId,
                                   linkIndexA=robLink2Idx['gripper_left_finger'],
                                   linkIndexB=objLink2Idx['objectLink'])
    fl.append(calculateForces(contact_l))

    contact_r = p.getContactPoints(bodyA=robId,
                                   bodyB=cylId,
                                   linkIndexA=robLink2Idx['gripper_right_finger'],
                                   linkIndexB=objLink2Idx['objectLink'])
    fr.append(calculateForces(contact_r))

    forces_l.append(np.mean(fl) / 100 + np.random.normal(0, 0.0077))
    forces_r.append(np.mean(fr) / 100 + np.random.normal(0, 0.0077))

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

    updatePlot()

    # c = p.getDebugVisualizerCamera()
    # print(f"{c[-2]}, {c[-4]}, {c[-3]}, {c[-1]}")

# use this to store force values to pickle
import os
import pickle

with open('{}/simClose.pkl'.format(os.environ['HOME']), 'wb') as f:
    pickle.dump({
        'forces_l': forces_l,
        'forces_r': forces_r
    }, f, protocol=2)

p.disconnect()
