import numpy as np

from tiago_rl.envs import TIAGoTactileEnv
from tiago_rl.misc import LoadCellVisualiser

# Environment setup
# ----------------------------

show_gui = True
env = TIAGoTactileEnv(show_gui=show_gui)

# Visualisation setup
# ----------------------------

vis = None
if show_gui:
    vis = LoadCellVisualiser()

# Trajectory sampling
# ----------------------------

waitSteps = 50
trajSteps = 140
gripper_qs = np.linspace(0.045, 0.00, num=trajSteps)
torso_qs = np.linspace(0, 0.05, num=trajSteps)

# Event Loop
# ----------------------------

for i in range(300):
    o, _, _, _ = env.step(env.desired_pos)

    # extract information from observations.
    # see GripperTactileEnv._get_obs() for reference.
    obs = o['observation']
    f = obs[-2:]

    if vis:
        vis.update_plot(f)
    elif i == 110:
        # test rendering if not showing GUI
        import matplotlib.pyplot as plt

        plt.imshow(env.render(height=1080, width=1920))
        plt.show()
