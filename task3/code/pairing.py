import numpy as np
from munkres import Munkres
import matplotlib.pyplot as plt

def pairing(a1, a2, b1, b2, k):
    value = []
    a1 = np.array(a1)
    b1 = np.array(b1)
    a2 = np.array(a2)
    b2 = np.array(b2)

    bt = b1.T
    corr1 = np.dot(a1, bt)
    (anum, bnum) = corr1.shape

    for a in range(anum):
        for b in range(bnum):
            for i in range(4):
                if a2[a][i] == b2[b][i]:
                    corr1[a][b] += 0.25
                if a2[a][i] == 0 and b2[b][i] == 1:
                    corr1[a][b] += 0.25

    exchange = 0
    #Munkres only works when anum < bnum
    if anum > bnum:
        corr1 = corr1.T
        exchange = 1
    corr = corr1.copy()

    indices = np.array(Munkres().compute(3-corr1))
    if exchange:
        indices = indices[:,::-1]
    for i in range(indices.shape[0]):
        if corr[indices[i][0]][indices[i][1]] < k:
            indices[i][1] = -1  #no match for a[indices[i][0]]
    return indices