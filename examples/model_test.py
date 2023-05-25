import numpy as np
np.set_printoptions(suppress=True, precision=3)

from tiago_rl.envs import GripperTactileEnv
from tiago_rl.misc.tactile_vis import LoadCellVisualiser

env = GripperTactileEnv(render_mode="human")
vis = LoadCellVisualiser(env)

for i in range(1000):
    if i % 50 == 0: 
        env.reset()
        vis.reset()

    action = 2*[0.005]
    obs, r, _, _, _ = env.step(action)
    vis.update_plot(action=action, reward=r)
env.close()
exit(0)


# viewer.vopt.frame = mj.mjtFrame.mjFRAME_BODY # mjFRAME_BODY | mjFRAME_WORLD | mjFRAME_CONTACT
# viewer.vopt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1