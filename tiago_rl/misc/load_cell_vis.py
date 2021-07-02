from pyqtgraph.Qt import QtGui, QtCore
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
        self.pl_rewa = self.win.addPlot(title="Current Reward")

        self.curve_raw_r = self.pl_raw.plot(pen='r')
        self.curve_raw_l = self.pl_raw.plot(pen='y')

        self.curve_curr_r = self.pl_curr.plot(pen='r')
        self.curve_curr_l = self.pl_curr.plot(pen='y')

        self.curve_succ = self.pl_succ.plot(pen='g')
        self.curve_rewa = self.pl_rewa.plot(pen='b')

        # buffers for plotted data
        self.raw_r = []
        self.raw_l = []

        self.curr_r = []
        self.curr_l = []

        self.succ = []
        self.rewa = []

    def update_plot(self, is_success, reward):

        # store new data
        self.raw_r.append(self.env.current_forces_raw[0])
        self.raw_l.append(self.env.current_forces_raw[1])

        self.curr_r.append(self.env.current_forces[0])
        self.curr_l.append(self.env.current_forces[1])

        self.succ.append(is_success)
        self.rewa.append(reward)

        # plot new data
        self.curve_raw_r.setData(self.raw_r)
        self.curve_raw_l.setData(self.raw_l)

        self.curve_curr_r.setData(self.curr_r)
        self.curve_curr_l.setData(self.curr_l)

        self.curve_succ.setData(self.succ)
        self.curve_rewa.setData(self.rewa)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

