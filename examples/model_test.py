import numpy as np
np.set_printoptions(suppress=True, precision=3)

from tiago_rl.envs.gripper_env import GripperEnv
from tiago_rl.misc.load_cell_vis import LoadCellVisualiser

env = GripperEnv(render_mode="human")
vis = LoadCellVisualiser(env)

for _ in range(1000):
    action = [0.00, 0.00]
    obs, r, _, _, _ = env.step(action)
    vis.update_plot(action=action, reward=r)
env.close()
exit(0)


# viewer.vopt.frame = mj.mjtFrame.mjFRAME_BODY # mjFRAME_BODY | mjFRAME_WORLD | mjFRAME_CONTACT
# viewer.vopt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1