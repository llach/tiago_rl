import time
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# set camera to view at scene
p.resetDebugVisualizerCamera(1.1823151111602783, 120.5228271484375, -68.42454528808594,
                             (-0.2751278877258301, -0.15310688316822052, -0.27969369292259216))

p.setGravity(0, 0, -10)

# load objects
planeId = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.01])
cylId = p.loadURDF("./assets/objects/cylinder.urdf", basePosition=[0.04, 0.02, 0.6])
robId = p.loadURDF("./assets/objects/box.urdf", basePosition=[0.0, 0.0, 0.27])

for _ in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)
p.resetSimulation()
