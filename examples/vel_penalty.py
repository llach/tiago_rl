import numpy as np
import matplotlib.pyplot as plt

def penv(v, vmax, alpha=6):
    vnorm = np.clip(np.abs(v), 0, vmax)/vmax
    return np.e**(alpha*((vnorm-1)))


xs = np.linspace(-0.2, 0.2, 500)
ys = [penv(x, 0.2) for x in xs]

plt.plot(xs, ys)
plt.show()

