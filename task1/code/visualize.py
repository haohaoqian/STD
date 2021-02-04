from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize(z, b, imgs):
    X = np.arange(0, 168)
    Y = np.arange(0, 168)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, z, cmap=cm.coolwarm)
    plt.show()
    plt.figure()
    for num in range(10):
        plt.subplot(1, 10, 1 + num)
        plt.imshow(imgs[num], cmap='gray')
    plt.show()