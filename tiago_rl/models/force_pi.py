import numpy as np
from enum import Enum

from tiago_rl import safe_rescale

class ControllerPhase(int, Enum):
    POSITION_CTRL=0
    FORCE_CLOSURE=1
    FORCE_CTRL=2

class ForcePI:

    def __init__(self, dt, fmax, fgoal, ftheta, Kp=1.9, Ki=3.1, k=1600, closing_vel=0.02, q_limits=[0.0, 0.045]):
        self.dt = dt
        self.fmax = fmax
        self.fgoal = fgoal
        self.ftheta = ftheta
        self.q_limits = q_limits

        # controller parameters
        self.k = k      # object stiffness
        self.Kp = Kp    # factor for p-part
        self.Ki = Ki    # factor for i-part
        self.closing_vel = closing_vel  # closing velocity during position control

        # reset (or initialize) controller state
        self.reset(self.fgoal)

    def reset(self, fgoal):
        self.fgoal = fgoal 

        # controller state
        self.joint_transition = [False, False]
        self.error_integral = 0.0

        self.phase = ControllerPhase.POSITION_CTRL

    def get_q(self, q, f_t):
        delta_qs = np.zeros_like(q)

        for i, f in enumerate(f_t):

            # phase I: closing
            if np.abs(f) < self.ftheta and not self.joint_transition[i]:
                delta_qs[i] = -self.closing_vel*self.dt*5
            # phase II: contact acquisition → waiting for force-closure
            elif not self.joint_transition[i]:
                self.phase = ControllerPhase.FORCE_CLOSURE
                self.joint_transition[i] = True
                print(f"joint {i} transition @ {q[i]}")

            # phase III: force control
            if np.all(self.joint_transition):
                if self.phase != ControllerPhase.FORCE_CTRL:
                    self.phase = ControllerPhase.FORCE_CTRL
                    print("transition to force control!")

                ''' from: https://github.com/llach/force_controller_core/blob/master/src/force_controller.cpp
                  // calculate new desired position
                  delta_F_ = target_force_ - *force_;
                  double delta_q_force = (delta_F_ / k_);

                  error_integral_ += delta_q_force * dt;
                  delta_q_ = K_p_ * delta_q_force + K_i_ * error_integral_;

                  // calculate new position and velocity
                  q_des_ = q - delta_q_;
                '''

                # force delta → position delta
                delta_f = self.fgoal - f
                delta_q = delta_f / self.k

                # integrate error TODO clip error integral?
                self.error_integral += delta_q * self.dt
                delta_q_ = self.Kp * delta_q + self.Ki * self.error_integral

                delta_qs[i] = -delta_q_

        # if self.phase == ControllerPhase.POSITION_CTRL:
        #     print("fc", delta_qs)
        # if self.phase == ControllerPhase.FORCE_CTRL:
        #     print("fc", delta_qs)

        return np.clip(q+delta_qs, *self.q_limits)
    
    def predict(self, obs, *args, **kwargs):
        """
        interface to be compatible with stable baselines' API
        """
        q_t = safe_rescale(obs[:2],  [-1,1], [0, 0.045])
        f_t = safe_rescale(obs[6:8], [-1,1], [0, self.fmax])

        qdes = self.get_q(q_t, f_t)
        return safe_rescale(qdes, [0, 0.045]), {}
