import platform
import numpy as np

from tiago_rl import safe_rescale
from tiago_rl.misc import VisBase, PlotItemWrapper as PIWrapper


class TactileVis(VisBase):

    def __init__(self, env):
        VisBase.__init__(
            self,
            env=env,
            title="Force Visualisation"
        )

        # create plots and curves
        self.plt_force = PIWrapper(self.win, title="Contact Forces", pens=["r", "y"], yrange=[-0.05, 1.2*env.fmax])
        self.plt_cntct = PIWrapper(self.win, title="In Contact", pens=["r", "y"], yrange=[-0.1, 1.1], ticks=[0,1])
        
        # draw lines at threshold and goal force
        self.draw_goal()
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

        self.plt_r = PIWrapper(self.win, title="r(t)", pens="g")
        self.plt_r_force = PIWrapper(self.win, title="r_force", pens="b")

        self.win.nextRow()

        self.plt_r_obj_prx = PIWrapper(self.win, title="r_obj_prx", pens="b", yrange=[-0.1, 2.2], ticks=[0,2])
        self.plt_r_qdot = PIWrapper(self.win, title="r_qdot", pens="r", yrange=[0.1, -2.2], ticks=[0,-2])

        self.all_plots = [
            self.plt_force, self.plt_cntct, 
            self.plt_pos, self.plt_vel, 
            self.plt_acc, self.plt_vobj, 
            self.plt_r, self.plt_r_force, 
            self.plt_r_obj_prx, self.plt_r_qdot
        ]

    def draw_goal(self):
        fgoal = self.env.fgoal
        self.plt_force.draw_line(
            name="fgoal",
            pos=fgoal,
            angle=0,
            pen={'color': "#00FF00"}
        )
        self.plt_force.draw_ticks([0, fgoal, self.env.fmax])

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
        self.plt_r_force.update(self.env.r_force)

        self.plt_r_obj_prx.update(self.env.r_obj_prx)
        self.plt_r_qdot.update(-self.env.r_qdot)

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

