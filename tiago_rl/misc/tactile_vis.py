import platform
import numpy as np
import pyqtgraph as pg

from PyQt6 import QtWidgets 
from tiago_rl import safe_rescale


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

        # create plots and curves
        self.pl_force = self.win.addPlot(title="Contact Forces")
        self.pl_cntct = self.win.addPlot(title="In Contact?")

        self.pl_cntct.setYRange(-0.09, 1.15)

        self._add_target_force_lines()

        # draw lines at threshold force
        for pl in [self.pl_force]:
            threshold_line = pg.InfiniteLine(
                pos=env.ftheta,
                angle=0
            )
            pl.addItem(threshold_line)

        self.win.nextRow()

        self.pl_q = self.win.addPlot(title="Joint Positions")
        self.pl_vel = self.win.addPlot(title="Joint Velocities")

        # joint position range with some padding
        self.pl_q.setYRange(0.05, -0.005)

        # set ticks at fully open, middle and fully closed
        ay = self.pl_q.getAxis('left')
        ticks = [0.045, 0.02, 0.0]
        ay.setTicks([[(v, str(v)) for v in ticks]])

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

        self.pl_obj_vel = self.win.addPlot(title="Object Velocity")
        self.pl_joint_acc = self.win.addPlot(title="Joint Accelerations")

        self.pl_obj_vel.setYRange(-0.02, 0.8)
        self.pl_obj_vel.showAxis('right')

        self.pl_joint_acc.setYRange(-6.2, 6.2)
        ay = self.pl_joint_acc.getAxis('left')
        ticks = [-4.0, 0, 4.0]
        ay.setTicks([[(v, str(v)) for v in ticks]])

        self.win.nextRow()

        self.pl_contact = self.win.addPlot(title="In Contact")
        self.pl_rewa = self.win.addPlot(title="Reward")

        self.win.nextRow()

        # self.pl_force_rew = self.win.addPlot(title="Force Reward")
        # self.pl_ovel_rew = self.win.addPlot(title="Obj.Vel. Reward")

        # self.pl_ovel_rew.setYRange(0.01 -1.0)

        self.all_plots = [self.pl_rewa, self.pl_contact, self.pl_obj_vel,
                          self.pl_joint_acc, self.pl_q, self.pl_vel,
                          self.pl_force, self.pl_cntct]

        self.curve_raw_r = self.pl_force.plot(pen='r')
        self.curve_raw_l = self.pl_force.plot(pen='y')

        self.curve_curr_r = self.pl_cntct.plot(pen='r')
        self.curve_curr_l = self.pl_cntct.plot(pen='y')

        self.curve_cont_r = self.pl_contact.plot(pen='g')
        self.curve_cont_l = self.pl_contact.plot(pen='b')

        self.curve_rewa = self.pl_rewa.plot(pen='b')

        self.curve_currv_r = self.pl_vel.plot(pen='r')
        self.curve_currv_l = self.pl_vel.plot(pen='y')

        self.curve_des_r = self.pl_q.plot(pen='c')
        self.curve_des_l = self.pl_q.plot(pen='b')

        self.curve_currq_r = self.pl_q.plot(pen='r')
        self.curve_currq_l = self.pl_q.plot(pen='y')

        self.curve_objv = self.pl_obj_vel.plot(pen='m')
        self.curve_objw = self.pl_obj_vel.plot(pen='b')

        self.curve_acc_r = self.pl_joint_acc.plot(pen='r')
        self.curve_acc_l = self.pl_joint_acc.plot(pen='y')

        # self.curve_force_rew = self.pl_force_rew.plot()
        # self.curve_ovel_rew = self.pl_ovel_rew.plot()

        # buffers for plotted data
        self.raw_r = []
        self.raw_l = []

        self.curr_r = []
        self.curr_l = []

        self.cont_r = []
        self.cont_l = []
        self.rewa = []

        self.des_r = []
        self.des_l = []

        self.currq_r = []
        self.currq_l = []
        self.vel_r = []
        self.vel_l = []

        self.rs = []

        self.objvs = []
        self.objws = []

        self.dv_force_r = []
        self.dv_force_l = []

        self.accel_r = []
        self.accel_l = []

        self.force_rew = []
        self.ovel_rew = []

    def _add_target_force_lines(self):
        tf = self.env.fgoal
        self.raw_target_line = pg.InfiniteLine(
            pos=tf,
            angle=0,
            pen={'color': "#00FF00"}
        )

        for pl, ln in zip([self.pl_force], [self.raw_target_line]):
            pl.addItem(ln)

            # always show target force in ticks
            ay = pl.getAxis('left')
            ticks = [0, tf]
            ay.setTicks([[(v, str(v)) for v in ticks]])
    
    def reset_target_force_lines(self):
        self.pl_force.removeItem(self.raw_target_line)

        self._add_target_force_lines()

    def reset(self):
        self.reset_target_force_lines()

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
        self.raw_l.append(self.env.forces[0])
        self.raw_r.append(self.env.forces[1])

        self.curr_l.append(self.env.in_contact[0])
        self.curr_r.append(self.env.in_contact[1])

        self.rs.append(reward)
        self.rewa.append(np.sum(self.rs))

        self.des_l.append(action[0])
        self.des_r.append(action[1])

        self.currq_l.append(self.env.q[0])
        self.currq_r.append(self.env.q[1])

        self.vel_l.append(self.env.qdot[0])
        self.vel_r.append(self.env.qdot[1])

        self.objvs.append(self.env.objv)
        self.objws.append(self.env.objw)

        self.cont_l.append(self.env.in_contact[0])
        self.cont_r.append(self.env.in_contact[1])

        if len(self.dv_force_l) == 0:
            self.dv_force_l.append(0)
            self.dv_force_r.append(0)
        else:
            self.dv_force_r.append((self.raw_r[-2]-self.raw_r[-1])/self.env.dt)
            self.dv_force_l.append((self.raw_l[-2]-self.raw_l[-1])/self.env.dt)

        self.accel_l.append(self.env.qacc[0])
        self.accel_r.append(self.env.qacc[1])

        # self.force_rew.append(self.env.force_rew)
        # self.ovel_rew.append(self.env.obj_vel_rew)

        # plot new data
        self.curve_raw_r.setData(self.raw_r)
        self.curve_raw_l.setData(self.raw_l)

        self.curve_curr_r.setData(self.curr_r)
        self.curve_curr_l.setData(self.curr_l)

        self.curve_cont_r.setData(self.cont_r)
        self.curve_cont_l.setData(self.cont_l)
        self.curve_rewa.setData(self.rs)

        self.curve_currq_r.setData(self.currq_r)
        self.curve_currq_l.setData(self.currq_l)

        self.curve_des_r.setData(self.des_r)
        self.curve_des_l.setData(self.des_l)

        self.curve_currv_r.setData(self.vel_r)
        self.curve_currv_l.setData(self.vel_l)

        self.curve_objv.setData(self.objvs)
        self.curve_objw.setData(self.objws)

        self.curve_acc_r.setData(self.accel_r)
        self.curve_acc_l.setData(self.accel_l)

        # self.curve_force_rew.setData(self.force_rew)
        # self.curve_ovel_rew.setData(self.ovel_rew)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

