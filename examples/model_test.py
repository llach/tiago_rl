import mujoco
import numpy as np
np.set_printoptions(suppress=True, precision=3)

from tiago_rl import safe_rescale
from tiago_rl.envs import GripperTactileEnv
from tiago_rl.misc import TactileVis

with_vis = 1

env = GripperTactileEnv(
    obj_pos_range=[0,0],
    **{"render_mode": "human"} if with_vis else {}
    )
vis = TactileVis(env) if with_vis else None

vdes = 0.15 # m/s
qdelta = vdes*0.1
qd = safe_rescale(qdelta, [0,0.045], [-1,1])

for i in range(1000):
    if i % 100 == 0: 
        env.reset()
        if vis: vis.reset()

    # fleft, _  = total_contact_force(env.model, env.data, "object", "left_finger_bb")
    # fright, _ = total_contact_force(env.model, env.data, "object", "right_finger_bb")

    # print(fleft[2])
    # print(fright[2])

    action = safe_rescale(
        np.clip(env.q - qdelta, 0, 0.045),
        [0, 0.045],
        [-1,1]
    )
    # action=[-1,-1]
    obs, r, _, _, _ = env.step(action)
    vis.update_plot(action=action, reward=r)
env.close()
exit(0)


# viewer.vopt.frame = mj.mjtFrame.mjFRAME_BODY # mjFRAME_BODY | mjFRAME_WORLD | mjFRAME_CONTACT
# viewer.vopt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1