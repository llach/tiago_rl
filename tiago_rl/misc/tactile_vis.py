import platform
import numpy as np
import pyqtgraph as pg

from PyQt6 import QtWidgets 
from tiago_rl import safe_rescale

from tiago_rl.misc import PlotItemWrapper as PIWrapper


class TactileVis:

    def __init__(self, env):
        # store reference to environment
        self.env = env

        # create QT Application
        self.app = QtWidgets.QApplication([])

        # configure window
        self.win = pg.GraphicsLayoutWidget(show=True, )
        self.win.resize(1000, 600)
        self.win.setWindowTitle('Force Visualisation')

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # update counter
        self.t = 0
        self.neps = 0

        # create plots and curves
        self.plt_force = PIWrapper(self.win, title="Contact Forces", pens=["r", "y"], yrange=[-0.05, 1.2*env.fmax])
        self.plt_cntct = PIWrapper(self.win, title="In Contact", pens=["r", "y"], yrange=[-0.05, 1.05], ticks=[0,1])
        
        # draw lines at threshold and goal force
        self.draw_fgoal()
        self.plt_force.draw_line(
                name="ftheta",
                pos=env.ftheta,
                angle=0
            )

        self.win.nextRow()

        self.plt_pos = PIWrapper(self.win, title="Joint Positions", pens=["r", "y", "c", "b"], yrange=[-0.005, 0.05], ticks=[0.045, 0.02, 0.0])

        vmax = self.env.vmax
        self.plt_vel = PIWrapper(self.win, title="Joint Velocities", pens=["r", "y"], yrange=[-1.3*vmax, 1.3*vmax], ticks=[-vmax, vmax])
        self.plt_vel.draw_line(
            name="upper_limit",
            pos=vmax,
            angle=0
        )
        self.plt_vel.draw_line(
            name="lower_limit",
            pos=-vmax,
            angle=0
        )

        self.win.nextRow()

        self.plt_acc = PIWrapper(self.win, title="Joint Accelerations", pens=["r", "y"], yrange=[-6.2, 6.2], ticks=[-4.0, 0, 4.0])
        self.plt_vobj = PIWrapper(self.win, title="Object Velocity", pens=["m", "b"], yrange=[-0.02, 0.8])

        self.win.nextRow()

        self.plt_r = PIWrapper(self.win, title="Reward", pens="g")

        self.all_plots = [self.plt_force, self.plt_cntct, self.plt_pos, self.plt_vel, self.plt_acc, self.plt_vobj, self.plt_r]

    def draw_fgoal(self):
        fgoal = self.env.fgoal
        self.plt_force.draw_line(
            name="fgoal",
            pos=fgoal,
            angle=0,
            pen={'color': "#00FF00"}
        )
        self.plt_force.draw_ticks([0, fgoal, self.env.fmax])

    def reset(self):
        self.draw_fgoal()

        for plot in self.all_plots:
            if self.t == 0: break
            plot.draw_line(
                name=f"ep{self.t}", 
                pos=self.t,
                pen={'color': "#D3D3D3", 'width': 1.5},
                    #  'style': QtCore.Qt.DotLine},
                angle=90
                )

    def update_plot(self, action, reward):
        self.t += 1
        action = safe_rescale(action, [-1,1], [0,0.045])

        # store new data
        self.plt_force.update(self.env.forces)
        self.plt_cntct.update(self.env.in_contact)

        self.plt_pos.update(np.concatenate([self.env.q, action]))
        self.plt_vel.update(self.env.qdot)

        self.plt_acc.update(self.env.qacc)
        self.plt_vobj.update([self.env.objv, self.env.objw])

        self.plt_r.update(reward)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

