import numpy as np
import matplotlib.pyplot as plt
from render import render

def evaluate(alpha,beta,s,X,Y,Z,n_s,lvector,gt):
    pred=render(alpha,beta,s,X,Y,Z,n_s,lvector)
    err = np.mean(np.mean(gt-pred,axis=-1),axis=-1)

    for num in range(7):
        plt.subplot(2,7,1+num)
        plt.imshow(gt[num],cmap='gray')
        plt.title('Mean Error={:.3f}'.format(err[num]))
        plt.subplot(2,7,8+num)
        plt.imshow(pred[num],cmap='gray')
    plt.show()