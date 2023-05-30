import numpy as np
import matplotlib.pyplot as plt

def penv(v, vmax, alpha=6):
    vnorm = np.clip(np.abs(v), 0, vmax)/vmax
    return vnorm
    return np.e**(alpha*((vnorm-1)))


xs = np.linspace(-0.2, 0.2, 500)
ys = [penv(x, 0.2) for x in xs]
ys2 = np.array([2**(np.abs(x)/0.2) for x in xs])
ys2 = ys2-np.min(ys2)

plt.plot(xs, ys)
plt.plot(xs, ys2)
plt.show()

