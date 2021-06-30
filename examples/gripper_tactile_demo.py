import numpy as np

from tiago_rl.envs import GripperTactileEnv

# ForcePlot setup
# ----------------------------

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import platform

app = QtGui.QApplication([])

win = pg.GraphicsLayoutWidget(show=True,)
win.resize(1000, 600)
win.setWindowTitle('Force Visualisation')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot(title="TA11 Scalar Contact Forces")

curve = p1.plot(pen='y')
curve2 = p1.plot(pen='r')
forces_l = []
forces_r = []

def update_plot():
    global curve, curve2, forces_l, forces_r, p1

    curve.setData(forces_l)
    curve2.setData(forces_r)

    if platform.system() == 'Linux':
        app.processEvents()

# Environment setup
# ----------------------------

show_gui = False
env = GripperTactileEnv(dt=1./240., show_gui=show_gui, force_noise_sigma=0.0077)

waitSteps = 50
trajSteps = 140
gripper_qs = np.linspace(0.045, 0.00, num=trajSteps)
torso_qs = np.linspace(0, 0.05, num=trajSteps)

# Event Loop
# ----------------------------
pos = [0.045, 0.045]

for i in range(300):
    if waitSteps < i < waitSteps+trajSteps:
        n = i-waitSteps
        o, _, _, _ = env.step(2*[gripper_qs[n]])
    else:
        o, _, _, _ = env.step(env.desired_pos)

    # extract information from observations.
    # see GripperTactileEnv._get_obs() for reference.
    obs = o['observation']
    f = obs[-2:]
    pos = obs[:2]

    if show_gui:
        forces_r.append(f[0])
        forces_l.append(f[1])

        curve.setData(forces_l)
        curve2.setData(forces_r)

        if platform.system() == 'Linux':
            app.processEvents()
    elif i == 110:
        # test rendering if not showing GUI
        import matplotlib.pyplot as plt

        plt.imshow(env.render(height=1080, width=1920))
        plt.show()