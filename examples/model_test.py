import numpy as np
np.set_printoptions(suppress=True, precision=5)

from tiago_rl import safe_rescale
from tiago_rl.envs import GripperTactileEnv, GripperPosEnv
from tiago_rl.misc import TactileVis, PosVis

with_vis = 1
steps  = 500
trials = 1

env = GripperTactileEnv(
    obj_pos_range=[-0.0,-0.0],
    **{"render_mode": "human", 'qinit_range': [0.025, 0.025]} if with_vis else {}
    )
vis = TactileVis(env) if with_vis else None

# env = GripperPosEnv(**{"render_mode": "human"} if with_vis else {})
# vis = PosVis(env) if with_vis else None

vdes = 0.02 # m/s
qdelta = vdes*env.dt
qd = safe_rescale(qdelta, [0,0.045], [-1,1])

for i in range(trials):
    env.reset()
    # env.set_goal(0.6)    
    if vis: vis.reset()

    for j in range(steps):
        # action = safe_rescale(
        #     np.clip(env.q - qdelta, 0, 0.045),
        #     [0, 0.045],
        #     [-1,1]
        # )
        action = [1,1]

        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)
env.close()