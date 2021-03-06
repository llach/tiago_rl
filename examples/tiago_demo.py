import argparse
import numpy as np

from tiago_rl.envs import TIAGoPALGripperEnv

# Parse CLI arguments
# ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--show_gui', default=False, action='store_true')
parser.add_argument('--no-show_gui', dest='show_gui', action='store_false')
args = parser.parse_args()

# Environment setup
# ----------------------------

show_gui = args.show_gui
env = TIAGoPALGripperEnv(show_gui=show_gui)

# Trajectory sampling
# ----------------------------

waitSteps = 50
trajSteps = 140
gripper_qs = np.linspace(0.045, 0.00, num=trajSteps)
torso_qs = np.linspace(0, 0.05, num=trajSteps)

# Event Loop
# ----------------------------

for i in range(300):
    if waitSteps < i < waitSteps + trajSteps:
        n = i - waitSteps

        new_state = env.create_desired_state({
            'gripper_right_finger_joint': gripper_qs[n],
            'gripper_left_finger_joint': gripper_qs[n],
        })
        o, _, _, _ = env.step(new_state)
    else:
        o, _, _, _ = env.step(env.desired_pos)

    # extract information from observations.
    # see GripperTactileEnv._get_obs() for reference.
    obs = o['observation']

    if i == 110 and not show_gui:
        # test rendering if not showing GUI
        import matplotlib.pyplot as plt

        plt.imshow(env.render(height=1080, width=1920))
        plt.show()
