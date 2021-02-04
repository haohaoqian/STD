import numpy as np
from PIL import Image

def rebuild(dir):
    images = np.zeros([7, 168, 168])
    lvector = np.zeros([7, 3])  # the direction of light
    for num in range(7):
        image = Image.open(dir + '/train/' + str(num + 1) + '.bmp')
        images[num] = np.asarray(image)
    for line in open(dir + '/train.txt'):
        i, ang1, ang2 = line.strip().split(",")
        i = int(i)
        ang1 = int(ang1)
        ang2 = int(ang2)
        lvector[i - 1] = (np.sin(np.pi * ang1 / 180) * np.cos(np.pi * ang2 / 180), np.sin(np.pi * ang2 / 180),
                          np.cos(np.pi * ang1 / 180) * np.cos(np.pi * ang2 / 180))
    lvector = -lvector
    vector = np.zeros([3, 168, 168])
    alpha = np.zeros([168, 168])
    b = np.zeros([3, 168, 168])
    for j in range(168):
        for k in range(168):
            b[:, j, k] = np.linalg.solve(np.dot(lvector.T, lvector), np.dot(lvector.T, images[:, j, k]))
            alpha[j, k] = np.linalg.norm(b[:, j, k], ord=2)
            temp = b[:, j, k] / alpha[j, k]
            if b[:, j, k][-1] > 0:
                vector[:, j, k] = -temp
            else:
                vector[:, j, k] = temp

    while True:
        pred = np.clip(np.einsum('ij,jkl->ikl', lvector, b), 0, 255)
        grad_b = (2 / 7) * np.sum(
            np.einsum('ijk,il->iljk', (pred - images), lvector) * ((pred > 0)[:, np.newaxis, :, :]), axis=0)
        b = b - grad_b
        if np.abs(np.max(grad_b)) <= 0.1:
            break

    return vector, b