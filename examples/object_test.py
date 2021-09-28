import numpy as np

from tiago_rl.envs import GripperTactileEnv


# Environment setup
# ----------------------------
initial_q = 0.025

show_gui = True
env = GripperTactileEnv(show_gui=show_gui,initial_state=np.array(2*[initial_q]), control_mode='pos')

# Trajectory sampling
# ----------------------------

steps = 100
gripper_qs = np.linspace(0.025, 0.00, num=steps)
torso_qs = np.linspace(0, 0.01, num=steps)

# Event Loop
# ----------------------------

for i in range(300):
    current_q, current_vel = env.get_state_dicts()
    current_forces = env.current_forces

    if i < steps:
        new_state = env.create_desired_state({
            'gripper_right_finger_joint': gripper_qs[i],
            'gripper_left_finger_joint': gripper_qs[i],
        })
    o, _, _, _ = env.step(new_state)
