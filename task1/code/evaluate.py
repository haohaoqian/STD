import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def evaluate(b, dir):
    gt = np.zeros([7, 168, 168])
    for num in range(7):
        gt[num] = np.asarray(Image.open(dir + '/train/' + str(num + 1) + '.bmp'))

    ang = np.zeros([7, 2])
    lvector = np.zeros([7, 3])  # the direction of light
    for line in open(dir + '/train.txt'):
        i, ang1, ang2 = line.strip().split(",")
        i = int(i)
        ang1 = int(ang1)
        ang2 = int(ang2)
        ang[i - 1] = (ang1, ang2)
        lvector[i - 1] = (np.sin(np.pi * ang1 / 180) * np.cos(np.pi * ang2 / 180), np.sin(np.pi * ang2 / 180),
                          np.cos(np.pi * ang1 / 180) * np.cos(np.pi * ang2 / 180))
    lvector = -lvector
    img = np.clip(np.einsum('ij,jkl->ikl', lvector, b), 0, 255)
    err = np.mean(np.mean(gt - img, axis=-1), axis=-1)
    plt.figure()
    for num in range(7):
        plt.subplot(2, 7, 1 + num)
        plt.imshow(gt[num], cmap='gray')
        plt.title('Ang1={} Ang2={}\nMean Error={:.3f}'.format(ang[num][0], ang[num][1], err[num]))
        plt.subplot(2, 7, 8 + num)
        plt.imshow(img[num], cmap='gray')
    plt.show()

    return img
