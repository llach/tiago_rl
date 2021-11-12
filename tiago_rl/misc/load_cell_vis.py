import numpy as np

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import platform

from tiago_rl.envs.bullet_robot_env import POS_CTRL, VEL_CTRL


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

        # update counter
        self.t = 0

        # create plots and curves
        self.pl_force = self.win.addPlot(title="Contact Forces")
        self.pl_cntct = self.win.addPlot(title="In Contact?")

        self.pl_cntct.setYRange(-0.05, 1.05)

        self._add_target_force_lines()

        # draw lines at threshold force
        for pl in [self.pl_force]:
            threshold_line = pg.InfiniteLine(
                pos=env.force_threshold,
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

        self.pl_obj_lin_vel = self.win.addPlot(title="Linear Object Velocity")
        self.pl_joint_acc = self.win.addPlot(title="Joint Accelerations")

        self.pl_obj_lin_vel.setYRange(-0.02, 0.2)
        self.pl_joint_acc.setYRange(-2.2*self.max_vel, 2.2*self.max_vel)

        self.win.nextRow()

        self.pl_succ = self.win.addPlot(title="Success State")
        self.pl_rewa = self.win.addPlot(title="Reward")

        self.pl_succ.setYRange(-0.2, 1.2)

        self.win.nextRow()

        self.pl_force_rew = self.win.addPlot(title="Force Reward")
        self.pl_ovel_rew = self.win.addPlot(title="Obj.Vel. Reward")

        self.pl_ovel_rew.setYRange(0.01, -1.0)

        self.all_plots = [self.pl_rewa, self.pl_succ, self.pl_obj_lin_vel,
                          self.pl_joint_acc, self.pl_q, self.pl_vel,
                          self.pl_force, self.pl_cntct]

        self.curve_raw_r = self.pl_force.plot(pen='r')
        self.curve_raw_l = self.pl_force.plot(pen='y')

        self.curve_curr_r = self.pl_cntct.plot(pen='r')
        self.curve_curr_l = self.pl_cntct.plot(pen='y')

        self.curve_succ = self.pl_succ.plot(pen='g')
        self.curve_rewa = self.pl_rewa.plot(pen='b')

        self.curve_currv_r = self.pl_vel.plot(pen='r')
        self.curve_currv_l = self.pl_vel.plot(pen='y')

        if env.control_mode == POS_CTRL:
            self.curve_des_r = self.pl_q.plot(pen='c')
            self.curve_des_l = self.pl_q.plot(pen='b')
        elif env.control_mode == VEL_CTRL:
            self.curve_des_r = self.pl_vel.plot(pen='c')
            self.curve_des_l = self.pl_vel.plot(pen='b')

        self.curve_currq_r = self.pl_q.plot(pen='g')
        self.curve_currq_l = self.pl_q.plot(pen='b')

        self.curve_obj_lin_vel = self.pl_obj_lin_vel.plot(pen='m')

        self.curve_acc_r = self.pl_joint_acc.plot(pen='r')
        self.curve_acc_l = self.pl_joint_acc.plot(pen='y')

        self.curve_force_rew = self.pl_force_rew.plot()
        self.curve_ovel_rew = self.pl_ovel_rew.plot()

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

        self.accel_r = []
        self.accel_l = []

        self.force_rew = []
        self.ovel_rew = []

    def _add_target_force_lines(self):
        tf = self.env.target_forces[0]
        self.raw_target_line = pg.InfiniteLine(
            pos=tf,
            angle=0
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
                    pen={'color': "#D3D3D3", 'width': 1.5,
                         'style': QtCore.Qt.DotLine},
                    angle=90
                )
            )

    def update_plot(self, is_success, reward):
        self.t += 1

        # store new data
        self.raw_r.append(self.env.current_forces_raw[0])
        self.raw_l.append(self.env.current_forces_raw[1])

        self.curr_r.append(self.env.in_contact[0])
        self.curr_l.append(self.env.in_contact[1])

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
            self.accel_l.append(0)
            self.accel_r.append(0)
        else:
            self.dv_force_r.append((self.raw_r[-2]-self.raw_r[-1])/self.env.dt)
            self.dv_force_l.append((self.raw_l[-2]-self.raw_l[-1])/self.env.dt)

        self.accel_r.append(self.env.current_acc[0])
        self.accel_l.append(self.env.current_acc[1])

        self.force_rew.append(self.env.force_rew)
        self.ovel_rew.append(self.env.obj_vel_rew)

        # plot new data
        self.curve_raw_r.setData(self.raw_r)
        self.curve_raw_l.setData(self.raw_l)

        self.curve_curr_r.setData(self.curr_r)
        self.curve_curr_l.setData(self.curr_l)

        self.curve_succ.setData(self.succ)
        self.curve_rewa.setData(self.rs)

        self.curve_currq_r.setData(self.currq_r)
        self.curve_currq_l.setData(self.currq_l)

        # self.curve_des_r.setData(self.des_r)
        # self.curve_des_l.setData(self.des_l)

        self.curve_currv_r.setData(self.vel_r)
        self.curve_currv_l.setData(self.vel_l)

        self.curve_obj_lin_vel.setData(self.obj_lin)

        self.curve_acc_r.setData(self.accel_r)
        self.curve_acc_l.setData(self.accel_l)

        self.curve_force_rew.setData(self.force_rew)
        self.curve_ovel_rew.setData(self.ovel_rew)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

