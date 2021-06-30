from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import platform


class LoadCellVisualiser:

    def __init__(self):
        # create QT Application
        self.app = QtGui.QApplication([])

        # configure window
        self.win = pg.GraphicsLayoutWidget(show=True, )
        self.win.resize(1000, 600)
        self.win.setWindowTitle('Force Visualisation')

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # create plot and curves
        self.pl = self.win.addPlot(title="TA11 Scalar Contact Forces")

        self.curve_right = self.pl.plot(pen='r')
        self.curve_left = self.pl.plot(pen='y')

        # buffers for force values
        self.forces_r = []
        self.forces_l = []

    def update_plot(self, forces):
        self.forces_r.append(forces[0])
        self.forces_l.append(forces[1])

        self.curve_right.setData(self.forces_r)
        self.curve_left.setData(self.forces_l)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

