<p align="center">
    <h1 align="center">Reinforcement Learning Environments for TIAGo</h1>
</p>

- [Installation](#installation)
- [Environments](#environments)
  - [TIAGoPALGripperEnv](#tiagopalgripperenv)
  - [TIAGoTactileEnv](#tiagotactileenv)
  - [GripperTactileEnv](#grippertactileenv)

This repository contains simulation environments for the TIAGo robot based on [pybullet](https://github.com/bulletphysics/bullet3).
They follow OpenAI's [gym](https://github.com/openai/gym) environment structure in order to by used in combination with 
reinforcement learning algorithms that follow the same convention (e.g. [baselines](https://github.com/openai/baselines)).


## Installation


1. Clone repository and change into the directory:
```
git clone https://github.com/llach/tiago_rl && cd tiago_rl
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Install the package itself:
```
pip install -e .
```
Using the pip flag `-e` leads to an editable installation of the project. This means, that changes to the source code are available directly without re-installing the project.

## Environments

Currently, this repository contains three different environments. 
All of them are based on OpenAI's [robotics environments](https://github.com/openai/gym/tree/master/gym/envs/robotics)
and thus similar to them in many regards.
As opposed to the standard robotics environments, this package offers tactile sensors also for mobile manipulators.

### [TIAGoPALGripperEnv](./tiago_rl/envs/tiago_env.py)
TIAGo with PAL Gripper using standard (non-sensorized) fingers.
![TIAGoPALGripperEnv](./images/tiago_default.png)

### [TIAGoTactileEnv](./tiago_rl/envs/load_cell_tactile_env.py)
 TIAGo with PAL Gripper using TA11 load cell sensors as fingers.
![TIAGoTactileEnv](./images/tiago_ta11.png)

### [GripperTactileEnv](./tiago_rl/envs/load_cell_tactile_env.py)
 Only PAL Gripper using TA11 load cell sensors as fingers.
Uses not as many complex collision bodies and thus may speed up policy learning for gripper-only policies.
![GripperTactileEnv](./images/gripper_ta11.png)
