import mujoco
import numpy as np
np.set_printoptions(suppress=True, precision=3)

import matplotlib.pyplot as plt

from tiago_rl import safe_rescale
from tiago_rl.envs import GripperTactileEnv
from tiago_rl.misc import TactileVis

with_vis = 0

steps  = 100

sidx   = 4
trials = 10
# vrange = [0.01, 0.99]
# values = np.linspace(*vrange, trials)

# values = np.arange(1,6)
values = 10*[2]
trials = len(values)
paramname="power"

env = GripperTactileEnv(
    obj_pos_range=[0,0],
    **{"render_mode": "human"} if with_vis else {}
    )
vis = TactileVis(env) if with_vis else None

# dertermine q delta for a certain velocity
vdes = 0.15 # m/s
qdelta = vdes*0.1
qd = safe_rescale(qdelta, [0,0.045], [-1,1])

forces = np.zeros((trials, steps))

for i in range(trials):
    solimp = env.SOLIMP
    solimp[sidx] = values[i]
    env.set_solver_parameters(solimp=solimp)
    env.reset()
    if vis: vis.reset()

    for j in range(steps):
        action = safe_rescale(
            np.clip(env.q - qdelta, 0, 0.045),
            [0, 0.045],
            [-1,1]
        )
        action=[-1,-1]
        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)

        forces[i,j]=env.forces[0]
env.close()
print("fmax", np.max(forces))

plt.figure(figsize=(9,6))
for v, ftraj in zip(values, forces):
    plt.plot(ftraj, label=f"{paramname}={v:.4f}")

plt.ylabel("force (left finger)")
plt.xlabel("time")
plt.title("contact force behavior for solimp changes")
plt.legend()

plt.tight_layout()
plt.show()