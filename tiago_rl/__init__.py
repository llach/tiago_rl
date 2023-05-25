import numpy as np

def safe_rescale(x, bounds1, bounds2):
    x = np.clip(x, *bounds1) # make sure x is within its interval
    
    low1, high1 = bounds1
    low2, high2 = bounds2
    return (((x - low1) * (high2 - low2)) / (high1 - low1)) + low2