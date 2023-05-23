import numpy as np
import matplotlib.pyplot as plt

from tiago_rl.envs.gripper_env import GripperEnv

def r(f, fgoal, band):
    fdelta = np.abs(fgoal - f)
    
    if fdelta>band: return 0 
    return np.e**(5*(((band-fdelta)/band)-1))

e = GripperEnv()

xs = np.linspace(0.0, 1.0, 500)
ys = [e._get_reward(x) for x in xs]

plt.plot(xs, ys)
plt.show()

