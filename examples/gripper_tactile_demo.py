from tiago_rl.envs import GripperTactileEnv



env = GripperTactileEnv(render=True)

for _ in range(300):
    env.step([0.02, 0.02])