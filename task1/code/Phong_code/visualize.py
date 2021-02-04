import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize(z,alpha):
    plt.figure()
    plt.imshow(alpha,cmap='gray')
    plt.show()

    X=np.arange(0,168)
    Y=np.arange(0,168)
    X,Y=np.meshgrid(X,Y)
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.plot_surface(X,Y,z)
    plt.show()