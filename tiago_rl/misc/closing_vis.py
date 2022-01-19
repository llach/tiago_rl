import numpy as np

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import platform

from tiago_rl.enums import ControlMode


class ClosingVisualiser:

    def __init__(self, env):
        # store reference to environment
        self.env = env

        # create QT Application
        self.app = QtGui.QApplication([])

        # configure window
        self.win = pg.GraphicsLayoutWidget(show=True, )
        self.win.resize(1000, 600)
        self.win.setWindowTitle('ClosingEnv Visualisation')

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # update counter
        self.t = 0

        self.pl_q = self.win.addPlot(title="Joint Positions")
        self.pl_vel = self.win.addPlot(title="Joint Velocities")

        # joint position range with some padding
        self.pl_q.setYRange(0.05, -0.005)

        # set ticks at fully open, middle and fully closed
        self._pos_ticks()

        # draw velocity maximums, set range and ticks
        if env.max_joint_velocities:
            self.max_vel = np.abs(list(env.max_joint_velocities.values())[0])

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

            self.pl_vel.setYRange(-self.max_vel * 1.1, self.max_vel * 1.1)

            ay = self.pl_vel.getAxis('left')
            ticks = [-self.max_vel, 0, self.max_vel]
            ay.setTicks([[(v, str(v)) for v in ticks]])

        self.win.nextRow()

        self.pl_error = self.win.addPlot(title="State Error")
        self.pl_rewa = self.win.addPlot(title="Reward")

        self.pl_error.setYRange(-0.045, 0.045)
        self.pl_rewa.setYRange(-1.1, 20.5)

        self.err_0_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen={'color': "#D3D3D3", 'width': 0.5, 'style': QtCore.Qt.DotLine}
        )
        self.pl_error.addItem(self.err_0_line)

        self.all_plots = [self.pl_q, self.pl_vel, self.pl_rewa]

        # curves
        self.curve_rewa = self.pl_rewa.plot(pen='b')

        self.curve_err_r = self.pl_error.plot(pen="r")
        self.curve_err_l = self.pl_error.plot(pen="y")

        self.curve_currq_r = self.pl_q.plot(pen='r')
        self.curve_currq_l = self.pl_q.plot(pen='y')

        self.curve_currv_r = self.pl_vel.plot(pen='r')
        self.curve_currv_l = self.pl_vel.plot(pen='y')

        if env.control_mode == ControlMode.POS_CTRL or env.control_mode == ControlMode.POS_DELTA_CTRL:
            self.curve_des_r = self.pl_q.plot(pen='c')
            self.curve_des_l = self.pl_q.plot(pen='b')
        elif env.control_mode == ControlMode.VEL_CTRL:
            self.curve_des_r = self.pl_vel.plot(pen='c')
            self.curve_des_l = self.pl_vel.plot(pen='b')

        # buffers for plotted data
        self.rs = []
        self.rewa = []

        self.des_r = []
        self.des_l = []

        self.currq_r = []
        self.currq_l = []

        self.vel_r = []
        self.vel_l = []

        self.err_r = []
        self.err_l = []


    def _pos_ticks(self):
        ay = self.pl_q.getAxis('left')
        ticks = [0.045, self.env.q_goal, 0.02, 0.0]
        ay.setTicks([[(v, str(v)) for v in ticks]])

    def _add_target_lines(self):
        tf = self.env.q_goal
        tfp = self.env.q_goal + self.env.goal_margin
        tfm = self.env.q_goal - self.env.goal_margin

        self.raw_target_line = pg.InfiniteLine(
            pos=tf,
            angle=0
        )
        p={'color': "#D3D3D3", 'width': 0.5, 'style': QtCore.Qt.DotLine}
        self.raw_target_linep = pg.InfiniteLine(
            pos=tfp,
            angle=0,
            pen=p,
        )
        self.raw_target_linem = pg.InfiniteLine(
            pos=tfm,
            angle=0,
            pen=p,
        )

        for ln in [self.raw_target_line, self.raw_target_linep, self.raw_target_linem]:
            self.pl_q.addItem(ln)

        # always show target force in ticks
        self._pos_ticks()
    
    def reset_target_lines(self):
        if hasattr(self, "raw_target_line"):
            self.pl_q.removeItem(self.raw_target_line)
            self.pl_q.removeItem(self.raw_target_linep)
            self.pl_q.removeItem(self.raw_target_linem)

        self._add_target_lines()

    def reset(self):
        self.reset_target_lines()

        for pl in self.all_plots:
            if self.t == 0:
                break
            pl.addItem(
                pg.InfiniteLine(
                    pos=self.t,
                    pen={'color': "#D3D3D3", 'width': 1.5,
                         'style': QtCore.Qt.DotLine},
                    angle=90
                )
            )

    def update_plot(self, is_success, reward):
        self.t += 1

        # store new data
        self.rs.append(reward)
        self.rewa.append(np.sum(self.rs))

        errors = self.env._goal_delta(self.env.current_pos)
        self.err_r.append(errors[0])
        self.err_l.append(errors[1])

        jq, jv = self.env.get_state_dicts()
        dq = self.env.get_desired_q_dict()

        if self.env.control_mode == ControlMode.POS_CTRL:
            self.des_r.append((dq['gripper_right_finger_joint']))
            self.des_l.append((dq['gripper_left_finger_joint']))
        elif self.env.control_mode == ControlMode.POS_DELTA_CTRL:
            self.des_r.append((dq['gripper_right_finger_joint']+jq['gripper_right_finger_joint']))
            self.des_l.append((dq['gripper_left_finger_joint']+jq['gripper_left_finger_joint']))

        self.currq_r.append((jq['gripper_right_finger_joint']))
        self.currq_l.append((jq['gripper_left_finger_joint']))

        self.vel_r.append((jv['gripper_right_finger_joint']))
        self.vel_l.append((jv['gripper_left_finger_joint']))

        # plot new data
        self.curve_rewa.setData(self.rs)

        self.curve_currq_r.setData(self.currq_r)
        self.curve_currq_l.setData(self.currq_l)

        self.curve_des_r.setData(self.des_r)
        self.curve_des_l.setData(self.des_l)

        self.curve_currv_r.setData(self.vel_r)
        self.curve_currv_l.setData(self.vel_l)

        self.curve_err_r.setData(self.err_r)
        self.curve_err_l.setData(self.err_l)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

