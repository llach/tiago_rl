import argparse
import numpy as np

from gym.wrappers import TimeLimit
from tiago_rl.envs import GripperTactileCloseEnv, TIAGoTactileEnv
from tiago_rl.misc import LoadCellVisualiser

# Parse CLI arguments
# ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', default='gripper_ta11', type=str,
                    help='environment type. defaults to gripper_ta11')
parser.add_argument('--show_gui', default=False, action='store_true')
parser.add_argument('--no-show_gui', dest='show_gui', action='store_false')
args = parser.parse_args()

# Environment setup
# ----------------------------

show_gui = args.show_gui
force_type = 'binary'
target_forces = np.array([1.0, 1.0])

if args.env == 'gripper_ta11':
    env = GripperTactileCloseEnv(show_gui=show_gui, force_type=force_type, target_forces=target_forces)
elif args.env == 'tiago_ta11':
    env = TIAGoTactileEnv(show_gui=show_gui, force_type=force_type, target_forces=target_forces)
else:
    print(f"Unknown environment {args.env}")
    exit(-1)

env = TimeLimit(env, max_episode_steps=150)

# Visualisation setup
# ----------------------------

vis = None
if show_gui:
    vis = LoadCellVisualiser(env)

# Trajectory sampling
# ----------------------------

waitSteps = 50
trajSteps = 140
gripper_qs = np.linspace(0.045, 0.00, num=trajSteps)
torso_qs = np.linspace(0, 0.05, num=trajSteps)

# Event Loop
# ----------------------------

for i in range(100):
    if waitSteps < i < waitSteps + trajSteps:
        n = i - waitSteps

        new_state = env.create_desired_state({
            'gripper_right_finger_joint': gripper_qs[n],
            'gripper_left_finger_joint': gripper_qs[n],
        })
        obs, reward, done, info = env.step(new_state)
    else:
        obs, reward, done, info = env.step(env.current_pos)

    # extract information from observations.
    # see GripperTactileEnv._get_obs() for reference.
    f = obs[-2:]

    if vis:
        vis.update_plot(is_success=info['is_success'], reward=reward)
    elif i == 110:
        # test rendering if not showing GUI
        import matplotlib.pyplot as plt

        plt.imshow(env.render(height=1080, width=1920))
        plt.show()

env.reset()

for i in range(100):
    if waitSteps < i < waitSteps + trajSteps:
        n = i - waitSteps

        new_state = env.create_desired_state({
            'gripper_right_finger_joint': gripper_qs[n],
            'gripper_left_finger_joint': gripper_qs[n],
        })
        obs, reward, done, info = env.step([np.nan, np.nan, np.nan])
    else:
        obs, reward, done, info = env.step(env.current_pos)

    if np.isnan(reward):
        print("OH NOOOO")

    # extract information from observations.
    # see GripperTactileEnv._get_obs() for reference.
    f = obs[-2:]

    if vis:
        vis.update_plot(is_success=info['is_success'], reward=reward)
    elif i == 110:
        # test rendering if not showing GUI
        import matplotlib.pyplot as plt

        plt.imshow(env.render(height=1080, width=1920))
        plt.show()