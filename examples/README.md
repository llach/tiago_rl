# TIAGo RL Demos

This folder contains a collection of demo scripts showcasing different environments and 
how to use them.

## Tactile Demo

TIAGo / Gripper demo using TA11 load cell sensors instead of non-sensorized fingers. Usage:

 `python tactile_demo.py`
 
Parameters:

* `--env (gripper_ta11|tiago_ta11)` 
choose between gripper-only and full robot environment. default is `gripper_ta11`

* `(--show_gui|--no-show_gui)`
if set, PyBullet visualisation and additional debug plots (forces, reward, ...) are shown.
  default is `False`
  
## TIAGo Demo

Demo with TIAGo using the PAL Gripper and non-sensorised fingers.

 `python tiago_demo.py`
 
Parameters:

* `(--show_gui|--no-show_gui)`
if set, PyBullet visualisation is shown. default is `False`