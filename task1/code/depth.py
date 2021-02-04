import numpy as np
import matlab.engine

def depth(vector, eng):
    vector = matlab.double(vector.tolist())
    z = eng.depth(vector)
    return np.array(z)