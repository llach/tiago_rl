import numpy as np

from tiago_rl.envs import GripperTactileEnv

env = GripperTactileEnv(render=True)

trajSteps = 140
gripper_qs = np.linspace(0.045, 0.00, num=trajSteps)
torso_qs = np.linspace(0, 0.05, num=trajSteps)

waitSteps = 50
for i in range(300):
    if waitSteps < i < waitSteps+trajSteps:
        n = i-waitSteps
        o = env.step(2*[gripper_qs[n]])