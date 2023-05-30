import pyqtgraph as pg

class PlotItemWrapper:

    def __init__(self, win: pg.GraphicsLayoutWidget, 
                 pens, title, yrange=None, ticks=None):
        
        self.data = []
        self.plot = win.addPlot(title=title)
        self.lines = {}

        self.curves = []
        self.curve_data = []

        # number of pens = number of curves
        if isinstance(pens, list) and len(pens)>1:
            for p in pens:
                self.curves.append(self.plot.plot(pen=p))
                self.curve_data.append([])
        elif isinstance(pens, str):
            self.curves = self.plot.plot(pen=pens)
        else:
            assert False, f"unsupported type for 'pens' {type(pens)}"

        if yrange: self.plot.setYRange(*yrange)
        if ticks:  self.draw_ticks(ticks)

    def _remove_line(self, name): 
        self.plot.removeItem(self.lines[name])
        self.lines.pop(name)

    def draw_line(self, name, **lnargs):
        if name in self.lines: self._remove_line(name)
        self.lines |= {
            name: pg.InfiniteLine(**lnargs)
        }
        self.plot.addItem(self.lines[name])

    def draw_ticks(self, ticks, axis="left"):
        ax = self.plot.getAxis(axis)
        ax.setTicks([[(v, str(v)) for v in ticks]])

    def update(self, v): 
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, bool):
            self.curve_data.append(v)
            self.curves.setData(self.curve_data)
        else:
            for x, curve, data in zip(v, self.curves, self.curve_data):
                data.append(x)
                curve.setData(data)