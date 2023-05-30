import platform
import numpy as np
import pyqtgraph as pg

from tiago_rl import safe_rescale
from tiago_rl.misc import VisBase, PlotItemWrapper as PIWrapper

class PosVis(VisBase):

    def __init__(self, env):

        VisBase.__init__(
            self,
            env=env,
            title="Position Control"
        )

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

        self.plt_r = PIWrapper(self.win, title="Reward", pens="g")
        self.plt_in_band = PIWrapper(self.win, title="In Band", pens="b")

        self.all_plots = [self.plt_pos, self.plt_vel, self.plt_r, self.plt_in_band]

    def draw_goal(self):
        qgoal = self.env.qgoal
        self.plt_pos.draw_line(
            name="qgoal",
            pos=qgoal,
            angle=0,
            pen={'color': "#00FF00"}
        )
        self.plt_pos.draw_ticks([0, qgoal, 0.045])

    def update_plot(self, action, reward):
        self.t += 1
        action = safe_rescale(action, [-1,1], [0,0.045])

        # store new data
        self.plt_pos.update(np.concatenate([self.env.q, action]))
        self.plt_vel.update(self.env.qdot)

        self.plt_r.update(reward)
        self.plt_in_band.update(self.env.in_band)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

