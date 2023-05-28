import mujoco
import numpy as np

def safe_rescale(x, bounds1, bounds2=[-1,1]):
    x = np.clip(x, *bounds1) # make sure x is within its interval
    
    low1, high1 = bounds1
    low2, high2 = bounds2
    return (((x - low1) * (high2 - low2)) / (high1 - low1)) + low2

def total_contact_force(model, data, g1, g2):
    force = np.zeros((3,))
    ncontacts = 0
    for i, c in enumerate(data.contact):
        name1 = data.geom(c.geom1).name
        name2 = data.geom(c.geom2).name

        if (g1==name1 and g2==name2) or (g1==name2 and g2==name1):
            ft = np.zeros((6,))
            mujoco.mj_contactForce(model, data, i, ft)
            force += ft[:3]
            ncontacts += 1
    return force, ncontacts