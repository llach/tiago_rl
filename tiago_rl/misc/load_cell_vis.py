import numpy as np

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

import platform


class LoadCellVisualiser:

    def __init__(self, env):
        # store reference to environment
        self.env = env

        # create QT Application
        self.app = QtGui.QApplication([])

        # configure window
        self.win = pg.GraphicsLayoutWidget(show=True, )
        self.win.resize(1000, 600)
        self.win.setWindowTitle('Force Visualisation')

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # create plots and curves
        self.pl_raw = self.win.addPlot(title="Raw Contact Forces")
        self.pl_curr = self.win.addPlot(title="Processed Contact Forces")

        self.win.nextRow()

        self.pl_succ = self.win.addPlot(title="Success State")
        self.pl_rewa = self.win.addPlot(title="Reward")

        self.win.nextRow()

        self.pl_q = self.win.addPlot(title="Joint Positions")
        self.pl_vel = self.win.addPlot(title="Joint Velocities")

        self.win.nextRow()

        self.pl_obj_lin_vel = self.win.addPlot(title="Linear Object Velocity")
        self.pl_obj_ang_vel = self.win.addPlot(title="Angular Object Velocity")

        self.curve_raw_r = self.pl_raw.plot(pen='r')
        self.curve_raw_l = self.pl_raw.plot(pen='y')

        self.curve_curr_r = self.pl_curr.plot(pen='r')
        self.curve_curr_l = self.pl_curr.plot(pen='y')

        self.curve_succ = self.pl_succ.plot(pen='g')
        self.curve_rewa = self.pl_rewa.plot(pen='b')

        self.curve_des_r = self.pl_q.plot(pen='r')
        self.curve_des_l = self.pl_q.plot(pen='y')

        self.curve_currq_r = self.pl_q.plot(pen='g')
        self.curve_currq_l = self.pl_q.plot(pen='b')

        self.curve_currv_r = self.pl_vel.plot(pen='g')
        self.curve_currv_l = self.pl_vel.plot(pen='b')

        self.curve_obj_lin_vel = self.pl_obj_lin_vel.plot(pen='m')
        self.curve_obj_ang_vel = self.pl_obj_ang_vel.plot(pen='w')

        self.threshold_line = pg.InfiniteLine(
            pos=env.force_threshold,
            angle=0
        )
        self.pl_raw.addItem(self.threshold_line)

        self.target_force_line = pg.InfiniteLine(
            pos=env.target_forces[0],
            angle=0
        )
        self.pl_raw.addItem(self.target_force_line)

        # buffers for plotted data
        self.raw_r = []
        self.raw_l = []

        self.curr_r = []
        self.curr_l = []

        self.succ = []
        self.rewa = []

        self.des_r = []
        self.des_l = []
        self.currq_r = []
        self.currq_l = []

        self.vel_r = []
        self.vel_l = []

        self.rs = []

        self.obj_lin = []
        self.obj_ang = []

        self.dv_force_r = []
        self.dv_force_l = []

    def reset(self, env):
        self.pl_raw.removeItem(self.target_force_line)
        self.target_force_line = pg.InfiniteLine(
            pos=env.target_forces[0],
            angle=0
        )
        self.pl_raw.addItem(self.target_force_line)

    def update_plot(self, is_success, reward):

        # store new data
        self.raw_r.append(self.env.current_forces_raw[0])
        self.raw_l.append(self.env.current_forces_raw[1])

        self.curr_r.append(self.env.current_forces[0])
        self.curr_l.append(self.env.current_forces[1])

        self.succ.append(is_success)
        self.rs.append(reward)
        self.rewa.append(np.sum(self.rs))

        jq, jv = self.env.get_state_dicts()
        dq = self.env.get_desired_q_dict()

        self.des_r.append((dq['gripper_right_finger_joint']))
        self.des_l.append((dq['gripper_left_finger_joint']))

        self.currq_r.append((jq['gripper_right_finger_joint']))
        self.currq_l.append((jq['gripper_left_finger_joint']))

        self.vel_r.append((jv['gripper_right_finger_joint']))
        self.vel_l.append((jv['gripper_left_finger_joint']))

        obj_v = self.env.get_object_velocity()

        self.obj_lin.append(np.linalg.norm(obj_v[0]))
        self.obj_ang.append(np.linalg.norm(obj_v[1]))

        if len(self.dv_force_l) == 0:
            self.dv_force_l.append(0)
            self.dv_force_r.append(0)
        else:
            self.dv_force_r.append((self.raw_r[-2]-self.raw_r[-1])/self.env.dt)
            self.dv_force_l.append((self.raw_l[-2]-self.raw_l[-1])/self.env.dt)

        # plot new data
        self.curve_raw_r.setData(self.raw_r)
        self.curve_raw_l.setData(self.raw_l)

        self.curve_curr_r.setData(self.curr_r)
        self.curve_curr_l.setData(self.curr_l)

        self.curve_succ.setData(self.succ)
        self.curve_rewa.setData(self.rs)

        self.curve_currq_r.setData(self.currq_r)
        self.curve_currq_l.setData(self.currq_l)

        self.curve_des_r.setData(self.des_r)
        self.curve_des_l.setData(self.des_l)

        self.curve_currv_r.setData(self.vel_r)
        self.curve_currv_l.setData(self.vel_l)

        self.curve_obj_lin_vel.setData(self.obj_lin)
        self.curve_obj_ang_vel.setData(self.obj_ang)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

