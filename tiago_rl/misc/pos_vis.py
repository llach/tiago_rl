import platform
import numpy as np
import pyqtgraph as pg

from PyQt6 import QtWidgets 
from tiago_rl import safe_rescale

class PosVis:

    def __init__(self, env):
        # store reference to environment
        self.env = env

        # create QT Application
        self.app = QtWidgets.QApplication([])

        # configure window
        self.win = pg.GraphicsLayoutWidget(show=True, )
        self.win.resize(1000, 600)
        self.win.setWindowTitle('Position Control')

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # update counter
        self.t = 0

        self.win.nextRow()

        self.pl_q = self.win.addPlot(title="Joint Positions")
        self.pl_vel = self.win.addPlot(title="Joint Velocities")

        # joint position range with some padding
        self.pl_q.setYRange(0.05, -0.005)

        # set ticks at fully open, middle and fully closed
        ay = self.pl_q.getAxis('left')
        ticks = [0.045, 0.02, 0.0]
        ay.setTicks([[(v, str(v)) for v in ticks]])

        self._add_target_q_lines()

        # draw velocity maxima, set range and ticks
        self.max_vel = 0.02 # m/s

        max_line = pg.InfiniteLine(
            pos=self.max_vel,
            angle=0
        )
        neg_max_line = pg.InfiniteLine(
            pos=-self.max_vel,
            angle=0
        )

        self.pl_vel.addItem(max_line)
        self.pl_vel.addItem(neg_max_line)

        self.pl_vel.setYRange(-self.max_vel * 1.3, self.max_vel * 1.3)

        ay = self.pl_vel.getAxis('left')
        ticks = [-self.max_vel, 0, self.max_vel]
        ay.setTicks([[(v, str(v)) for v in ticks]])

        self.win.nextRow()

        self.pl_band = self.win.addPlot(title="In Band")
        self.pl_rewa = self.win.addPlot(title="Reward")

        self.all_plots = [self.pl_rewa, self.pl_q, self.pl_vel, self.pl_band]

        self.curve_rewa = self.pl_rewa.plot(pen='b')

        self.curve_currv_r = self.pl_vel.plot(pen='r')
        self.curve_currv_l = self.pl_vel.plot(pen='y')

        self.curve_des_r = self.pl_q.plot(pen='c')
        self.curve_des_l = self.pl_q.plot(pen='b')

        self.curve_currq_r = self.pl_q.plot(pen='r')
        self.curve_currq_l = self.pl_q.plot(pen='y')

        self.curve_band = self.pl_band.plot(pen='y')

        # buffers for plotted data
        self.rs = []
        self.rewa = []
        self.in_band = []

        self.des_r = []
        self.des_l = []

        self.currq_r = []
        self.currq_l = []

        self.vel_r = []
        self.vel_l = []

    def _add_target_q_lines(self):
        tf = self.env.qgoal
        self.raw_target_line = pg.InfiniteLine(
            pos=tf,
            angle=0,
            pen={'color': "#00FF00"}
        )

        for pl, ln in zip([self.pl_q], [self.raw_target_line]):
            pl.addItem(ln)

            # always show target force in ticks
            ay = pl.getAxis('left')
            ticks = [0, tf]
            ay.setTicks([[(v, str(v)) for v in ticks]])
    
    def reset_target_q_lines(self):
        self.pl_q.removeItem(self.raw_target_line)
        self._add_target_q_lines()

    def reset(self):
        self.reset_target_q_lines()

        for pl in self.all_plots:
            if self.t == 0:
                break
            pl.addItem(
                pg.InfiniteLine(
                    pos=self.t,
                    pen={'color': "#D3D3D3", 'width': 1.5},
                        #  'style': QtCore.Qt.DotLine},
                    angle=90
                )
            )

    def update_plot(self, action, reward):
        self.t += 1
        action = safe_rescale(action, [-1,1], [0,0.045])

        # store new data
        self.rs.append(reward)
        self.rewa.append(np.sum(self.rs))
        self.in_band.append(self.env.in_band)

        self.des_l.append(action[0])
        self.des_r.append(action[1])

        self.currq_l.append(self.env.q[0])
        self.currq_r.append(self.env.q[1])

        self.vel_l.append(self.env.qdot[0])
        self.vel_r.append(self.env.qdot[1])

        # plot new data
        self.curve_rewa.setData(self.rs)
        self.curve_band.setData(self.in_band)

        self.curve_currq_r.setData(self.currq_r)
        self.curve_currq_l.setData(self.currq_l)

        self.curve_des_r.setData(self.des_r)
        self.curve_des_l.setData(self.des_l)

        self.curve_currv_r.setData(self.vel_r)
        self.curve_currv_l.setData(self.vel_l)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

